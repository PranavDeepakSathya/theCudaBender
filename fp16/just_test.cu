#include "../atoms/all.cuh"
constexpr int mma_m = 16;
constexpr int mma_n = 8; 
constexpr int mma_k = 16;
constexpr int m = mma_m;
constexpr int n = mma_n;
constexpr int k = mma_k; 
typedef __half nvfloat16;
constexpr int grid_size = 1;
constexpr int block_size = 32;
constexpr uint32_t As_bytes = m*k*sizeof(nvfloat16);
constexpr uint32_t Bs_bytes = k*n*sizeof(nvfloat16);
constexpr uint32_t Cs_bytes = m*n*sizeof(nvfloat16);
constexpr uint32_t shared_bytes = As_bytes + Bs_bytes + Cs_bytes + 4*1024;
namespace wa = warp_function; 

__global__ void matmul(__grid_constant__ const CUtensorMap a_map,
                       __grid_constant__ const CUtensorMap b_map,
                      __grid_constant__ const CUtensorMap c_map)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t As = smem;
  uint32_t Bs = As + As_bytes;
  uint32_t Cs = Bs + Bs_bytes; 
  uint32_t bar = Cs + Cs_bytes; 
  int l = threadIdx.x; 
  if (l ==0)  mbarrier_init(bar, 32); 
  __syncthreads(); 
  if (l == 0)
  {
    mbarrier_arrive_expect_tx(bar, As_bytes + Bs_bytes);
    cp_async_bulk_tensor_2d(As, &a_map, 0,0, bar);
    cp_async_bulk_tensor_2d(Bs, &b_map, 0,0, bar);
  }
  else mbarrier_arrive(bar);

  mbarrier_wait_parity(bar,0); 
  uint32_t ra[4];
  uint32_t rb[2];
  uint32_t rc[2] = {0.0};

  uint32_t a_ld_addr = As + ((((l%16)*k) + (8*(l/16)))*sizeof(nvfloat16));
  uint32_t b_ld_addr = Bs + (((l % 8) * k + (8 * (l / 8))) * sizeof(nvfloat16));
  wa::ldmatrix_m8n8_x4_b16(ra, a_ld_addr);
  wa::ldmatrix_m8n8_x2_b16(rb, b_ld_addr);
  wa::mma_m16n8k16_row_col_f16_f16(rc,ra,rb);
  uint32_t c_st_addr = Cs + ((l%16)*n*sizeof(nvfloat16));
  wa::stmatrix_m8n8_x2_b16(rc,c_st_addr);

  cp_async_bulk_tensor_2d_store(&c_map, 0,0, Cs);
  cp_async_commit_group();
  cp_async_wait_group<0>();


}


int main()
{
  NaiveTensor<nvfloat16>A({m,k}, Layout::ROW_MAJOR);
  NaiveTensor<nvfloat16>B({k,n}, Layout::COL_MAJOR);
  NaiveTensor<nvfloat16>C({m,n}, Layout::ROW_MAJOR);
  A.allocate(); B.allocate(); C.allocate(); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); B.to_device(); C.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nvfloat16>::create_2d_row_major(A.d_ptr, {m,k},{m,k});
  CUtensorMap b_map = TmaDescriptor<nvfloat16>::create_2d_col_major(B.d_ptr, {k,n},{k,n});
  CUtensorMap c_map = TmaDescriptor<nvfloat16>::create_2d_row_major(C.d_ptr, {m,n},{m,n});

  NaiveLauncher launcher(grid_size,1,block_size,shared_bytes);
  launcher.launch(matmul, a_map,b_map,c_map); 
  cudaDeviceSynchronize();
  C.to_host(); 
  NaiveTensor<nvfloat16> C_ref({m,n}, Layout::ROW_MAJOR);
  C_ref.allocate();
  C_ref.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  printf("A[0]=%f\n", __half2float(A.h_ptr[0]));
  printf("B[0]=%f\n", __half2float(B.h_ptr[0]));

  // host-side reference matmul
  for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {

          float acc = 0.0f;

          for (int kk = 0; kk < k; ++kk) {
              float a = __half2float(A.h_ptr[i*k + kk]);
              float b = __half2float(B.h_ptr[kk + j*k]);;
              acc += a * b;
          }

          C_ref.h_ptr[i*n + j] = __float2half(acc);
      }
  }
  C_ref.to_device();
  C_ref.to_host();

  printf("\n===== A =====\n");
  A.pretty_print();

  printf("\n===== B =====\n");
  B.pretty_print();

  printf("\n===== C (Tensor Core FP16 accumulate) =====\n");
  C.pretty_print();

  printf("\n===== C_ref (Host reference FP32 accumulate → FP16) =====\n");
  C_ref.pretty_print();
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  int mismatch_count = 0;

  for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {

          float tc = __half2float(C.h_ptr[i*n + j]);
          float ref = __half2float(C_ref.h_ptr[i*n + j]);

          float abs_err = fabsf(tc - ref);
          float rel_err = fabsf(tc - ref) / (fabsf(ref) + 1e-6f);

          if (abs_err > max_abs_err) max_abs_err = abs_err;
          if (rel_err > max_rel_err) max_rel_err = rel_err;

          if (abs_err > 5e-2f) {   // FP16 accumulate tolerance
              mismatch_count++;
          }
      }
  }

  printf("\n===== VERIFICATION =====\n");
  printf("Max abs error: %f\n", max_abs_err);
  printf("Max rel error: %f\n", max_rel_err);
  printf("Mismatches (>5e-2): %d / %d\n", mismatch_count, m*n);

}
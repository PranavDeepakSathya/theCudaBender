#include "../atoms/all.cuh"
constexpr int m = 16; 
constexpr int k = 16; 
constexpr int n = 8; 
constexpr int block_size = 32; 
constexpr int grid_size = 1; 
constexpr int As_bytes = m*k*sizeof(nv_bfloat16);
constexpr int Bs_bytes = n*k*sizeof(nv_bfloat16);
namespace wa = warp_function; 

__global__ void matmul(__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap b_map, float*C)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t As = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); 
  uint32_t Bs = As + As_bytes; 
  uint32_t bar = Bs + Bs_bytes; 
  int l = threadIdx.x; 
  uint32_t ra[4]; 
  uint32_t rb[2];
  float rc[4] = {0.0};
  if (l==0) mbarrier_init(bar,32); 
  __syncthreads(); 
  if (l==0)
  {
    cp_async_bulk_tensor_2d(As,&a_map,0,0,bar); 
    cp_async_bulk_tensor_2d(Bs,&b_map,0,0,bar); 
    mbarrier_arrive_expect_tx(bar, As_bytes + Bs_bytes); 
  }
  else mbarrier_arrive(bar); 
  mbarrier_wait_parity(bar,0); 
  //from this point onrward, treat shared memory As if it is (k,n) col major, as that is equivalent to 
  //(n,k) being row major 
  uint32_t a_ld_addr = As + (((l%16)*k + 8*(l/16))*sizeof(nv_bfloat16));
  uint32_t b_ld_addr = Bs + (((l%8)*k + 8*(l/8))*sizeof(nv_bfloat16));
  wa::ldmatrix_m8n8_x4_b16(ra, a_ld_addr);
  wa::ldmatrix_m8n8_x2_b16(rb, b_ld_addr); 
  wa::mma_m16n8k16_row_col_f32_bf16(rc,ra,rb);
  float2* C2 = reinterpret_cast<float2*>(C); 
  int lane_row = l/4; 
  int lane_col = 2*(l%4); 
  int ldc2 = n/2;
  int C_row = lane_row;
  int C_col = (lane_col)/2;
  float2 v0 = {rc[0], rc[1]}; 
  float2 v1 = {rc[2], rc[3]}; 
  C2[(C_row)*ldc2 + (C_col)] = v0; 
  C2[(C_row+8)*ldc2 + (C_col)] = v1;


}
int main()
{
  NaiveTensor<nv_bfloat16>A({m,k}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>B({n,k}, Layout::ROW_MAJOR); 
  NaiveTensor<float>C({m,n}, Layout::ROW_MAJOR); 
  A.allocate(); B.allocate(); C.allocate(); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); B.to_device(); C.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr, {m,k},{m,k});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(B.d_ptr, {n,k},{n,k});

  NaiveLauncher launcher(grid_size, 1, block_size, As_bytes + Bs_bytes + 16); 
  launcher.launch(matmul, a_map, b_map, C.d_ptr); 
  cudaDeviceSynchronize(); 
  C.to_host(); 
    // ---- host reference ----
  std::vector<float> C_ref(m * n, 0.0f);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; kk++) {
        float a = __bfloat162float(A.h_ptr[i*k + kk]);
        float b = __bfloat162float(B.h_ptr[j*k + kk]);
        acc += a * b;
      }
      C_ref[i*n + j] = acc;
    }
  }

  // ---- compare ----
  float max_err = 0.0f;
  for (int i = 0; i < m*n; i++) {
    float c_gpu = C.h_ptr[i];
    float diff = fabsf(c_gpu - C_ref[i]);
    if (diff > max_err) max_err = diff;
  }

  printf("Max abs error: %f\n", max_err);


}
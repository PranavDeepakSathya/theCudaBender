#include "../atoms/all.cuh"

constexpr int mma_m = 16; 
constexpr int mma_n = 8;
constexpr int mma_k = 16; 
constexpr uint32_t As_bytes = mma_m*mma_k*sizeof(nv_bfloat16); 
constexpr uint32_t Bs_bytes = mma_k*mma_n*sizeof(nv_bfloat16); 
constexpr uint32_t shared_allocate_bytes = As_bytes + Bs_bytes + (128*4); 

constexpr int M = mma_m; 
constexpr int N = mma_n; 
constexpr int BM = mma_m; 
constexpr int BN = mma_n; 
constexpr int K = mma_k; 
constexpr int BK = mma_k; 
constexpr int block_size = 32; 
constexpr int grid_size = 1;

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__global__ void one_warp_one_matmul (__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  float* C)

{

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  nv_bfloat16* As = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);
  smem = align128(smem);
  nv_bfloat16* Bs = (nv_bfloat16*)smem;

  uint32_t smem_base_a = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_b = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));
  uint32_t ra0,ra1,ra2,ra3; 
  uint32_t rb0,rb1; 
  float rc0=0.0,rc1=0.0,rc2=0.0,rc3=0.0; 
  int l = threadIdx.x;

  __shared__ barrier bar; 

  if (l == 0)
  {
    init(&bar,blockDim.x); 
  }

  __syncthreads();

  barrier::arrival_token token; 
  
  if (is_elected())
  {
     

    int32_t coords_A[2] = {0,0};
    int32_t coords_B[2] = {0,0}; 

    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, As_bytes + Bs_bytes);
  }
  else
  {
    token = bar.arrive();
  }

  bar.wait(std::move(token)); 

  int a_lane_row = l % 16;
  int a_lane_col = (l / 16) * 8;

  int b_lane_row = (l / 8) * 8;
  int b_lane_col = l % 8;

  uint32_t a_ld_addr = (a_lane_col +(a_lane_row*BK))*sizeof(nv_bfloat16) + (smem_base_a);
  uint32_t b_ld_addr = (b_lane_row +(b_lane_col*BK))*sizeof(nv_bfloat16) + (smem_base_b);

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(ra0), "=r"(ra1), "=r"(ra2), "=r"(ra3)
    : "r"(a_ld_addr)
  );
      
      

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
    : "=r"(rb0), "=r"(rb1) 
    : "r"(b_ld_addr)
  );

    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};\n"
    : "=f"(rc0), "=f"(rc1), "=f"(rc2), "=f"(rc3)
    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3),
      "r"(rb0), "r"(rb1),
      "f"(rc0), "f"(rc1), "f"(rc2), "f"(rc3)
  );

  float2* C2 = reinterpret_cast<float2*>(C); 
  float2 v0 = {rc0,rc1}; 
  float2 v1 = {rc2, rc3}; 

  
  int lane_row = l/4; 
  int lane_col = l%4; 

  int ldc2 = N/2; 
  C2[(lane_row+0)*ldc2 + lane_col] = v0; 
  C2[(lane_row+8)*ldc2 + lane_col] = v1; 
}

int main()
{
  NaiveTensor<nv_bfloat16> A({M,K}, Layout::ROW_MAJOR);
  NaiveTensor<nv_bfloat16> B({K,N}, Layout::COL_MAJOR); 
  NaiveTensor<float>C({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<float> C_ref({M, N}, Layout::ROW_MAJOR);

  A.allocate();
  B.allocate();
  C.allocate();
  C_ref.allocate(); 

  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  C_ref.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

  A.to_device();
  B.to_device();
  C.to_device();
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{M,K},{BM,BK});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {K,N}, {BK,BN});
  NaiveLauncher launcher(grid_size, 1, block_size,shared_allocate_bytes);

  //correctness launch
  launcher.launch(one_warp_one_matmul, a_map, b_map, C.d_ptr);
  cudaDeviceSynchronize(); 
  C.to_host(); 
  

  // host reference matmul
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float acc = 0.0f;

      for (int k = 0; k < K; ++k)
      {
        // A is row-major: A[i][k]
        float a = __bfloat162float(A.h_ptr[i*K + k]);

        // B is column-major: B[k][j]
        float b = __bfloat162float(B.h_ptr[j*K + k]);

        acc += a * b;
      }

      C_ref.h_ptr[i*N + j] = acc;
    }
  }
  
C_ref.to_device();

  float max_err = 0.0f;

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float ref = C_ref.h_ptr[i*N + j];
      float gpu = C.h_ptr[i*N + j];

      float err = fabsf(ref - gpu);
      max_err = fmaxf(max_err, err);

      if (err > 1e-2f)
      {
        printf("Mismatch at (%d,%d): ref=%f gpu=%f\n",
              i, j, ref, gpu);
        goto done;
      }
    }
  }

  done:
  printf("Max error: %f\n", max_err);


printf(" \n ================= C_actual ============ \n");
C_ref.pretty_print(); 
printf("\n ================ C_computed ============= \n");
C.pretty_print();

}


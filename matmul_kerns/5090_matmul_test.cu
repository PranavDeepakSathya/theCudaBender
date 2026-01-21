#include "../atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;


constexpr int M = 1024; 
constexpr int K = 1024; 
constexpr int N = 1024; 
constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 
constexpr int warps_m = 2; 
constexpr int warps_n = 4; 
constexpr int BM = mma_m*warps_m; 
constexpr int BN = mma_n*warps_n; 
constexpr int k_iters = 4; 
constexpr int BK = mma_k*k_iters; 
constexpr int grid_size = (M*N)/(BM*BN); 
constexpr int block_size = warps_m*warps_n*32; 
constexpr int GM = M/BM; 
constexpr int GN = N/BN; 

void cpu_gemm_ref(
    const NaiveTensor<nv_bfloat16>& A,
    const NaiveTensor<nv_bfloat16>& B,
    NaiveTensor<float>& C_ref
);

__global__ void matmul (__grid_constant__ const CUtensorMap gA, __grid_constant__ const CUtensorMap gB,
  NaiveTensor<float>::DeviceView C)

{
  __shared__ alignas(128) nv_bfloat16 As[BM*BK]; 
  __shared__ alignas(128) nv_bfloat16 Bs[BK*BN]; 
  int t = threadIdx.x;

  int w = t/32;
  int l = t%32;
  int b = blockIdx.x; 

  int gm = b / GN; 
  int gn = b % GN; 
  
  int block_m_start = gm*BM; 
  int block_n_start = gn*BN; 
  nv_bfloat162 ra[4]; 
  nv_bfloat162 rb[2]; 
  float rc[4];

  int wm = w / warps_n; 
  int wn = w % warps_n; 
  int warp_m_start = wm*mma_m; 
  int warp_n_start = wn*mma_n;


  __shared__ barrier bar; 
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  barrier::arrival_token token; 

  for (int block_k_start = 0; block_k_start < K; block_k_start += BK)
  {
    
    if (is_elected())
    {
      int32_t coords_A[2] = {block_k_start, block_m_start};
      int32_t coords_B[2] = {block_k_start, block_n_start}; 

      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
      token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As)+sizeof(Bs));
    }
    else
    {
      token = bar.arrive();
    }
    bar.wait(std::move(token));

    for (int warp_k_start = 0; warp_k_start < BK; warp_k_start += mma_k)
    {
      

      int a_lane_group_16x16_id = l/16; 
      int a_lane_row_id = l % 16;
      int a_lane_group_offset = a_lane_group_16x16_id*8; 
      int a_row_idx = warp_m_start + a_lane_row_id; 
      int a_col_idx = warp_k_start + a_lane_group_offset; 
      int a_flat_offset = a_col_idx + (a_row_idx*BK); 
      
      uint32_t smem_base_a = static_cast<uint32_t>(__cvta_generic_to_shared(As));
      uint32_t a_ld_addr = smem_base_a + (a_flat_offset * sizeof(nv_bfloat16));
      uint32_t* a_reg_ptr = reinterpret_cast<uint32_t*>(ra);
  
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a_reg_ptr[0]), "=r"(a_reg_ptr[1]), "=r"(a_reg_ptr[2]), "=r"(a_reg_ptr[3])
        : "r"(a_ld_addr)
      );
      
      int b_lane_group_16x8_id = l/8; 
      int b_lane_col_id = l % 8; 
      int b_lane_group_offset = b_lane_group_16x8_id*8; 
      int b_row_idx = warp_k_start + b_lane_group_offset;
      int b_col_idx = warp_n_start + b_lane_col_id; 
      int b_flat_offset = b_row_idx + (b_col_idx*BK); 

      uint32_t smem_base_b = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));
      uint32_t b_ld_addr = smem_base_b + (b_flat_offset * sizeof(nv_bfloat16)); 
      uint32_t* b_reg_ptr = reinterpret_cast<uint32_t*>(rb);
      
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(b_reg_ptr[0]), "=r"(b_reg_ptr[1]) 
        : "r"(b_ld_addr)
      );
      
      __syncthreads();
      float* c_reg_ptr = reinterpret_cast<float*>(rc);

      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c_reg_ptr[0]), "=f"(c_reg_ptr[1]), "=f"(c_reg_ptr[2]), "=f"(c_reg_ptr[3])
        : "r"(a_reg_ptr[0]), "r"(a_reg_ptr[1]), "r"(a_reg_ptr[2]), "r"(a_reg_ptr[3]),
          "r"(b_reg_ptr[0]), "r"(b_reg_ptr[1]),
          "f"(c_reg_ptr[0]), "f"(c_reg_ptr[1]), "f"(c_reg_ptr[2]), "f"(c_reg_ptr[3])
    );

    }
    
  }

  // if (blockIdx.x == 0 && threadIdx.x == 0)
  // {
  //   printf("%f, %f, %f, %f", rc[0],rc[1],rc[2],rc[3]);
  // }  
  __syncthreads();

  int lane_m = l/4; 
  int lane_n = 2*(l%4); 
  int c_total_offset_at_warp_gran_m = block_m_start + warp_m_start; 
  int c_total_offset_at_warp_gran_n = block_n_start + warp_n_start; 
  int store_ro_m = c_total_offset_at_warp_gran_m + lane_m; 
  int store_ro_n = c_total_offset_at_warp_gran_n + lane_n; 
  int store_r1_m = c_total_offset_at_warp_gran_m + lane_m; 
  int store_r1_n = c_total_offset_at_warp_gran_n + lane_n + 1; 
  int store_r2_m = c_total_offset_at_warp_gran_m + lane_m + 8; 
  int store_r2_n = c_total_offset_at_warp_gran_n + lane_n; 
  int store_r3_m = c_total_offset_at_warp_gran_m + lane_m + 8; 
  int store_r3_n = c_total_offset_at_warp_gran_n + lane_n + 1;   

  C.get(store_ro_m, store_ro_n) = rc[0];
  C.get(store_r1_m, store_r1_n) = rc[1];
  C.get(store_r2_m, store_r2_n) = rc[2];
  C.get(store_r3_m, store_r3_n) = rc[3];

}



int main()
{
  NaiveTensor<nv_bfloat16> A({M,K}, Layout::ROW_MAJOR);
  NaiveTensor<nv_bfloat16> B({K,N}, Layout::COL_MAJOR); 
  NaiveTensor<float>C({M,N}, Layout::ROW_MAJOR); 
  A.allocate();
  B.allocate();
  C.allocate();
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  A.to_device();
  B.to_device();
  C.to_device();
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{M,K},{BM,BK});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {K,N}, {BK,BN});
  auto c_view = C.get_device_view();
  for (int i = 0; i < 3; ++i) {
    matmul<<<grid_size, block_size>>>(a_map, b_map, c_view);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  matmul<<<grid_size, block_size>>>(a_map, b_map, c_view);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  printf("matmul kernel time: %.4f ms\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  C.to_host();
  // GEMM FLOPs = 2 * M * N * K
  double flops = 2.0 * double(M) * double(N) * double(K);

  // ms -> seconds
  double seconds = ms * 1e-3;

  // FLOP/s
  double tflops = flops / seconds / 1e12;

  printf("FLOPs: %.3e\n", flops);
  printf("Time:  %.4f ms\n", ms);
  printf("TFLOP/s: %.2f\n", tflops);

  // NaiveTensor<float> C_ref({M, N}, Layout::ROW_MAJOR);
  // C_ref.allocate();
  // cpu_gemm_ref(A, B, C_ref);

  // float max_abs_err = 0.0f;
  // float max_rel_err = 0.0f;

  // for (int i = 0; i < M; ++i)
  // {
  //     for (int j = 0; j < N; ++j)
  //     {
  //         float ref = C_ref.get_host(i, j);
  //         float gpu = C.get_host(i, j);

  //         float abs_err = fabs(ref - gpu);
  //         float rel_err = abs_err / (fabs(ref) + 1e-6f);

  //         max_abs_err = fmax(max_abs_err, abs_err);
  //         max_rel_err = fmax(max_rel_err, rel_err);
  //     }
  // }

  // printf("max abs error = %e\n", max_abs_err);
  // printf("max rel error = %e\n", max_rel_err);


}




void cpu_gemm_ref(
    const NaiveTensor<nv_bfloat16>& A,
    const NaiveTensor<nv_bfloat16>& B,
    NaiveTensor<float>& C_ref
)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                float a = __bfloat162float(A.get_host(i, k));
                float b = __bfloat162float(B.get_host(k, j));
                acc += a * b;
            }
            C_ref.get_host(i, j) = acc;
        }
    }
}

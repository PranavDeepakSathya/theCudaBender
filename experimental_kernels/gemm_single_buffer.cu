#include "../atoms/all.cuh"
#include "canon_config_single_buffer.cuh"

namespace ptx = cuda::ptx; 
using barrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    float* C,
    int M, int N, int K);

void verify_result(
const NaiveTensor<float>& C,
const NaiveTensor<float>& C_ref,
int M, int N,
float abs_thresh,
int max_print);

template <class cfg>
void benchmark_matmul(
    const char* name,
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev,
    int warmup_iters = 20,
    int bench_iters  = 2000);


template <class cfg>
__global__ void matmul (const __grid_constant__ CUtensorMap a_map, 
  const __grid_constant__ CUtensorMap b_map, float *C)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As;
  nv_bfloat16* Bs; 
  As = reinterpret_cast<nv_bfloat16*>(smem_raw); 
  Bs = As + cfg::SMEM_A_ELEMS; 

  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int b = blockIdx.x; 
  int block_m0 = (b / cfg::GRID_N)*cfg::BLOCK_M; 
  int block_n0 = (b % cfg::GRID_N)*cfg::BLOCK_N; 
  int warp_m0 = (w / cfg::WARPS_PER_BLOCK_N)*cfg::WARP_M; 
  int warp_n0 = (w % cfg::WARPS_PER_BLOCK_N)*cfg::WARP_N; 

  int a_lane_row = (l%16);
  int a_lane_col = 8*(l/16);
  int b_lane_col = (l%8); 
  int b_lane_row = 8*(l/8); 


  uint32_t ra[cfg::ACC_PER_WARP_M][4]; 
  uint32_t rb[cfg::ACC_PER_WARP_N][2];
  float rc[cfg::ACC_PER_WARP_M][cfg::ACC_PER_WARP_N][4] = {0.0};


  __shared__ barrier bar; 

  // if (t == 0)
  // {
  //   init(&bar, blockDim.x);
  // }
  

  barrier::arrival_token token;

  for (int bk_iter = 0; bk_iter < cfg::BLOCK_K_ITERS; bk_iter++)
  {
    if (t == 0)
    {
      init(&bar, blockDim.x);
    }
    __syncthreads();

    if (t == 0)
    {
      int32_t A_coords[2] = {bk_iter*cfg::BLOCK_K, block_m0};
      int32_t B_coords[2] = {bk_iter*cfg::BLOCK_K, block_n0};

      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        As, &a_map, A_coords, cuda::device::barrier_native_handle(bar));

      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        Bs, &b_map, B_coords, cuda::device::barrier_native_handle(bar));

      token = cuda::device::barrier_arrive_tx(bar, 1, cfg::SMEM_A_BYTES + cfg::SMEM_B_BYTES);
    }
    else
    {
      token = bar.arrive();
    }

    bar.wait(std::move(token));
    asm volatile("fence.proxy.async.shared::cta;");


    for (int wk_idx = 0; wk_idx < cfg::MMA_K_ITERS_PER_WARP; wk_idx++)
    {
      for (int wm_idx = 0; wm_idx < cfg::ACC_PER_WARP_M; wm_idx++)
      {
        int a_shared_row = warp_m0 + (wm_idx*cfg::MMA_M) + a_lane_row;
        int a_shared_col = (wk_idx*cfg::MMA_K) + a_lane_col; 
        int a_shared_offset = (a_shared_row*cfg::BLOCK_K) + a_shared_col; 
        nv_bfloat16* a_ptr = As + a_shared_offset; 
        uint32_t a_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(a_ptr));

        warp_atom::ldmatrix_m8n8_x4_b16(ra[wm_idx][0], ra[wm_idx][1], ra[wm_idx][2], ra[wm_idx][3],a_smem_addr);

      }

      for (int wn_idx = 0; wn_idx < cfg::ACC_PER_WARP_N; wn_idx++)
      {
        int b_shared_row = (wk_idx*cfg::MMA_K) + b_lane_row; 
        int b_shared_col = warp_n0 + (wn_idx*cfg::MMA_N) + b_lane_col; 
        int b_shared_offset = (b_shared_col*cfg::BLOCK_K) + b_shared_row; 
        nv_bfloat16* b_ptr = Bs + b_shared_offset; 
        uint32_t b_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(b_ptr)); 
        warp_atom::ldmatrix_m8n8_x2_b16(rb[wn_idx][0], rb[wn_idx][1], b_smem_addr);
      }

      for (int wm_idx = 0; wm_idx < cfg::ACC_PER_WARP_M; wm_idx++)
      {
        for (int wn_idx = 0; wn_idx < cfg::ACC_PER_WARP_N; wn_idx++)
        {
           warp_atom::mma_m16n8k16_row_col_f32_bf16(
            rc[wm_idx][wn_idx][0], rc[wm_idx][wn_idx][1], rc[wm_idx][wn_idx][2], rc[wm_idx][wn_idx][3],
            ra[wm_idx][0],ra[wm_idx][1], ra[wm_idx][2], ra[wm_idx][3],
            rb[wn_idx][0], rb[wn_idx][1]
           );
        }
      }
    }


  }
  __syncthreads(); 

  float2* C2 = reinterpret_cast<float2*>(C); 
  int lane_row = l/4; 
  int lane_col = 2*(l%4); 
  int ldc2 = cfg::N/2;

   for (int wm_idx = 0; wm_idx < cfg::ACC_PER_WARP_M; wm_idx++)
      {
        for (int wn_idx = 0; wn_idx < cfg::ACC_PER_WARP_N; wn_idx++)
        {
          int C_row = block_m0 + warp_m0 + (wm_idx*cfg::MMA_M) + lane_row;
          int C_col = (block_n0 + warp_n0 + (wn_idx*cfg::MMA_N) + lane_col)/2;
          float2 v0 = {rc[wm_idx][wn_idx][0],rc[wm_idx][wn_idx][1]}; 
          float2 v1 = {rc[wm_idx][wn_idx][2],rc[wm_idx][wn_idx][3]};
          C2[(C_row)*ldc2 + (C_col)] = v0; 
          C2[(C_row+8)*ldc2 + (C_col)] = v1; 
        }
      }


}

int main()
{
  using Cfg = sm120_single_buffer_gemm_config;

  NaiveTensor<nv_bfloat16> A({Cfg::M, Cfg::K}, Layout::ROW_MAJOR);
  NaiveTensor<nv_bfloat16> B({Cfg::K, Cfg::N}, Layout::COL_MAJOR);
  NaiveTensor<float>       C({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);
  NaiveTensor<float>       C_ref({Cfg::M, Cfg::N}, Layout::ROW_MAJOR); 


  A.allocate(); B.allocate(); C.allocate(); C_ref.allocate();

  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

  A.to_device();
  B.to_device();
  C.to_device();
  C_ref.to_device(); 


  CUtensorMap a_map =
      TmaDescriptor<nv_bfloat16>::create_2d_row_major(
          A.d_ptr, {Cfg::M, Cfg::K}, {Cfg::BLOCK_M, Cfg::BLOCK_K});

  CUtensorMap b_map =
      TmaDescriptor<nv_bfloat16>::create_2d_col_major(
          B.d_ptr, {Cfg::K, Cfg::N}, {Cfg::BLOCK_K, Cfg::BLOCK_N});

  NaiveLauncher launcher(Cfg::GRID_BLOCKS, 1, Cfg::BLOCK_THREADS, Cfg::SMEM_BYTES_TOTAL);

  launcher.launch(matmul<Cfg>, a_map, b_map, C.d_ptr);

  cudaDeviceSynchronize();
  C.to_host();

  printf("Launch complete.\n");

    dim3 block(16, 16);
  dim3 grid(
      (Cfg::N + block.x - 1) / block.x,
      (Cfg::M + block.y - 1) / block.y
  );

  naive_gemm_ref<<<grid, block>>>(
      A.d_ptr,
      B.d_ptr,
      C_ref.d_ptr,
      Cfg::M, Cfg::N, Cfg::K
  );

  cudaDeviceSynchronize();
  
  C_ref.to_host();

  verify_result(C, C_ref, Cfg::M, Cfg::N,1e-1,20);
}




__global__ void naive_gemm_ref(
    const nv_bfloat16* A,   // [M, K] row-major
    const nv_bfloat16* B,   // [K, N] column-major
    float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k)
    {
        // A(i,k)
        float a = __bfloat162float(A[row * K + k]);

        // B(k,j) — COLUMN MAJOR
        float b = __bfloat162float(B[col * K + k]);

        acc += a * b;
    }

    C[row * N + col] = acc;
}

void verify_result(
    const NaiveTensor<float>& C,
    const NaiveTensor<float>& C_ref,
    int M, int N,
    float abs_thresh,
    int max_print)
{
    int bad_count = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    const float eps = 1e-6f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float ref = C_ref.h_ptr[i * N + j];
            float gpu = C.h_ptr[i * N + j];

            float abs_err = fabsf(ref - gpu);
            float rel_err = abs_err / fmaxf(fabsf(ref), eps);

            max_abs_err = fmaxf(max_abs_err, abs_err);
            max_rel_err = fmaxf(max_rel_err, rel_err);

            if (abs_err > abs_thresh)
            {
                if (bad_count < max_print)
                {
                    printf(
                        "BAD (%d,%d) ref=%f gpu=%f abs=%e rel=%e\n",
                        i, j, ref, gpu, abs_err, rel_err
                    );
                }
                bad_count++;
            }
        }
    }

    printf("\n==== Verification ====\n");
    printf("Max abs error : %e\n", max_abs_err);
    printf("Max rel error : %e\n", max_rel_err);
    printf("Bad elements  : %d (abs > %.2e)\n",
           bad_count, abs_thresh);

    if (bad_count == 0)
        printf("✅ PASSED\n");
    else
        printf("❌ FAILED\n");
}

template <class cfg>
void benchmark_matmul(
    const char* name,
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev,
    int warmup_iters,
    int bench_iters)
{
    for (int i = 0; i < warmup_iters; i++)
        launcher.launch(matmul<cfg>, a_map, b_map, C_dev);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < bench_iters; i++)
        launcher.launch(matmul<cfg>, a_map, b_map, C_dev);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_ms = total_ms / bench_iters;
    double avg_s  = avg_ms * 1e-3;

    double flops  = 2.0 * double(cfg::M) * double(cfg::N) * double(cfg::K);
    double tflops = flops / (avg_s * 1e12);

    printf("\n========== Benchmark ==========\n");
    printf("Kernel   : %s\n", name);
    printf("Shape    : M=%d N=%d K=%d\n", cfg::M, cfg::N, cfg::K);
    printf("Avg time : %.4f ms\n", avg_ms);
    printf("TFLOP/s  : %.2f\n", tflops);
    printf("================================\n");
}

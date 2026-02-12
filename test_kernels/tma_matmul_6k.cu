#include "../atoms/all.cuh"
#include "sm_120_matmul_config_6000.cuh"

namespace ptx = cuda::ptx; 

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
__global__ void matmul
          (
            __grid_constant__ const CUtensorMap a_map, 
            __grid_constant__ const CUtensorMap b_map,
            float* C
          )
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As[cfg::bk_stages];
  nv_bfloat16* Bs[cfg::bk_stages]; 
  uint8_t* ptr = smem_raw;

  #pragma unroll 
  for (int stage = 0; stage < cfg::bk_stages; stage++)
  {
    As[stage] = alloc<nv_bfloat16,1024>(ptr, cfg::BM*cfg::BK); 
  }

  #pragma unroll 
  for (int stage = 0; stage < cfg::bk_stages; stage++)
  {
    Bs[stage] = alloc<nv_bfloat16,1024>(ptr, cfg::BK*cfg::BN); 
  }

  uint64_t* empty = alloc<uint64_t, 8>(ptr, cfg::bk_stages);
  uint64_t* full = alloc<uint64_t, 8>(ptr, cfg::bk_stages);
  uint64_t empty_tokens[cfg::bk_stages];
  uint64_t full_tokens[cfg::bk_stages];
  
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int b = blockIdx.x; 

  if (t == 0)
  { 
    #pragma unroll
    for (int stage = 0; stage < cfg::bk_stages; stage++)
    {
      ptx::mbarrier_init(&empty[stage],cfg::block_size);
      ptx::mbarrier_init(&full[stage],cfg::block_size);
  
    }
  }
  __syncthreads();

  int b_start_m = (b/cfg::GN)*cfg::BM; 
  int b_start_n = (b%cfg::GN)*cfg::BN;
  int w_start_m = (w/cfg::wpb_n)*cfg::WM;
  int w_start_n = (w%cfg::wpb_n)*cfg::WN; 

  if (w == cfg::producer_warp_id)
  { 
    #pragma unroll
    for (int bk_idx = 0; bk_idx < cfg::bk_iters; bk_idx++)
    {
      int stage = bk_idx % cfg::bk_stages; 
      int32_t A_coords[2] = {bk_idx*cfg::BK, b_start_m};
      int32_t B_coords[2] = {bk_idx*cfg::BK, b_start_n}; 
      empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]);
      while(!ptx::mbarrier_try_wait(&empty[stage], empty_tokens[stage])); 
      asm volatile("fence.proxy.async.shared::cta;");
      if (l == 0)
      {
        ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
          As[stage], &a_map, A_coords, &full[stage]);

        ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
          Bs[stage], &b_map, B_coords, &full[stage]);

        full_tokens[stage] = ptx::mbarrier_arrive_expect_tx(
          ptx::sem_release, 
          ptx::scope_cta, 
          ptx::space_shared,
          &full[stage],
          cfg::As_bytes + cfg::Bs_bytes
        );
      }
      else
      {
        full_tokens[stage] = ptx::mbarrier_arrive(&full[stage]);
      }
    }
  }
  else
  {
    uint32_t ra[cfg::apw_m*4];
    uint32_t rb[cfg::apw_n*2];
    float rc[cfg::apw_m*cfg::apw_n*4] = {0.0}; 
    #pragma unroll
    for (int stage = 0; stage < cfg::bk_stages; stage++)
    {
      empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]); 

    }
    #pragma unroll
    for (int bk_idx = 0; bk_idx < cfg::bk_iters; bk_idx++)
    {
      int stage = bk_idx % cfg::bk_stages; 
      full_tokens[stage] = ptx::mbarrier_arrive(&full[stage]);
      while(!ptx::mbarrier_try_wait(&full[stage], full_tokens[stage]));

      #pragma unroll
      for (int wk_idx = 0; wk_idx < cfg::wk_iters; wk_idx++)
      {
        #pragma unroll
        for (int wm_idx = 0; wm_idx < cfg::apw_m; wm_idx++)
        {
          int a_ld_shared_offset = (w_start_m + (wm_idx)*cfg::mma_m + (l%16))*cfg::BK + ((wk_idx)*cfg::mma_k + 8*(l/16));
          nv_bfloat16* a_ptr = As[stage] + a_ld_shared_offset;
          uint32_t a_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(a_ptr));
          int a_reg_start = wm_idx*4; 
          warp_atom::ldmatrix_m8n8_x4_b16(ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3], a_smem_addr);

        }
        #pragma unroll
        for (int wn_idx = 0; wn_idx < cfg::apw_n; wn_idx++)
        {
          int b_ld_shared_offset = (w_start_n + (wn_idx)*cfg::mma_n + (l%8))*cfg::BK + ((wk_idx)*cfg::mma_k + 8*(l/8));
          nv_bfloat16* b_ptr = Bs[stage] + b_ld_shared_offset; 
          uint32_t b_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(b_ptr));
          int b_reg_start = wn_idx*2; 
          warp_atom::ldmatrix_m8n8_x2_b16(rb[b_reg_start + 0], rb[b_reg_start + 1],b_smem_addr);
        }
        #pragma unroll
        for (int wm_idx = 0; wm_idx < cfg::apw_m; wm_idx++)
        {
          #pragma unroll
          for (int wn_idx = 0; wn_idx < cfg::apw_n; wn_idx++)
          {
            int a_reg_start = wm_idx*4; 
            int b_reg_start = wn_idx*2; 
            int c_reg_start = 4*(wn_idx + cfg::apw_n*wm_idx); 
            warp_atom::mma_m16n8k16_row_col_f32_bf16(
              rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3],
              ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3],
              rb[b_reg_start + 0], rb[b_reg_start + 1]

            );
          }
        }

      }

      empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]); 
    }

    float2* C2 = reinterpret_cast<float2*>(C); 
    int lane_row = l/4; 
    int lane_col = 2*(l%4); 
    int ldc2 = cfg::N/2;

    #pragma unroll
    for (int wm_idx = 0; wm_idx < cfg::apw_m; wm_idx++)
    {
      #pragma unroll
      for (int wn_idx = 0; wn_idx < cfg::apw_n; wn_idx++)
      {
        int C_row = b_start_m + w_start_m + (wm_idx*cfg::mma_m) + lane_row;
        int C_col = (b_start_n + w_start_n + (wn_idx*cfg::mma_n) + lane_col)/2;
        int c_reg_start = 4*(wn_idx + cfg::apw_n*wm_idx); 
        float2 v0 = {rc[c_reg_start + 0], rc[c_reg_start + 1]}; 
        float2 v1 = {rc[c_reg_start + 2], rc[c_reg_start + 3]}; 
        C2[(C_row)*ldc2 + (C_col)] = v0; 
        C2[(C_row+8)*ldc2 + (C_col)] = v1;
      }
    }

  }

  

}


int main() 
{
  using Cfg = Sm120_BF16_Gemm_Config;

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
          A.d_ptr, {Cfg::M, Cfg::K}, {Cfg::BM, Cfg::BK});

  CUtensorMap b_map =
      TmaDescriptor<nv_bfloat16>::create_2d_col_major(
          B.d_ptr, {Cfg::K, Cfg::N}, {Cfg::BK, Cfg::BN});

  NaiveLauncher launcher(Cfg::grid_size, 1, Cfg::block_size, Cfg::shared_bytes);

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

  benchmark_matmul<Cfg>(
    "sm120 bf16 matmul",
    launcher,
    a_map,
    b_map,
    C.d_ptr
  );

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

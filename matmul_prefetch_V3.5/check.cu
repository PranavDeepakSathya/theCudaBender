#include "../atoms/all.cuh"
#include "config.cuh"
#include "smem_allocator.cuh"

namespace ptx = cuda::ptx;
namespace wa = warp_function;

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C
)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  SmemAllocator<Cfg> smem(smem_raw);
  int b = blockIdx.x;
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t % 32; 
  int block_start_m, block_start_n; 
  tile_sched::block_swizzle<Cfg::group_m, Cfg::group_n, Cfg::blocks_per_group, Cfg::G_outer_M, Cfg::G_outer_N, Cfg::BM,Cfg::BN>(b,block_start_m,block_start_n);
  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN;
  int C_row_start = block_start_m + warp_start_m + (l/4);
  int C_col_start = block_start_n + warp_start_n + 2*(l%4);
  uint32_t a_ld_base = ((warp_start_m + (l%16))*Cfg::BK + (8*(l/16)))*sizeof(nv_bfloat16);
  uint32_t b_ld_base = ((warp_start_n + (l%8))*Cfg::BK + (8*(l/8)))*sizeof(nv_bfloat16);
  float2* C2 = reinterpret_cast<float2*>(C);
  int ldc2 = Cfg::N/2;
  uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_stages][4];
  uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_stages][2];
  float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0};

  if (t == 0)
  {
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
    {
      mbarrier_init(smem.full(stage),32); 
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  auto TMA_load_A_B = [&](int bk_idx, int stage)
  {
    if (l == 0)
    {
      cp_async_bulk_tensor_2d(smem.A(stage), &a_map, bk_idx*Cfg::BK, block_start_m, smem.full(stage));
      cp_async_bulk_tensor_2d(smem.B(stage), &b_map, bk_idx*Cfg::BK, block_start_n, smem.full(stage));
      mbarrier_arrive_expect_tx(smem.full(stage), Cfg::As_bytes + Cfg::Bs_bytes);
    }
    else 
    {
      mbarrier_arrive(smem.full(stage));
    }
  };

  auto ldm_A_k = [&](uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_stages][4], int k, int stage, int wk_stage)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      uint32_t a_ld_addr = smem.A(stage) + compact_swizzle<Cfg::swizzle_num>(a_ld_base + ((m*Cfg::mma_m*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x4_b16(ra[m][wk_stage], a_ld_addr);
    }

  };

  auto ldm_B_k = [&](uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_stages][2], int k, int stage, int wk_stage)
  {
    #pragma unroll 
    for (int n = 0; n < Cfg:: acc_per_warp_n; n++)
    {
      uint32_t b_ld_addr = smem.B(stage) + compact_swizzle<Cfg::swizzle_num>(b_ld_base + ((n*Cfg::mma_n*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x2_b16(rb[n][wk_stage], b_ld_addr);
    }

  };  

  auto mma_k = [&](float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4],
                  uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_stages][4],
                  uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_stages][2],
                  int k,int wk_stage)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      #pragma unroll
      for (int n = 0; n < Cfg::acc_per_warp_n; n++)
      {
        wa::mma_m16n8k16_row_col_f32_bf16(
            rc[m][n],
            ra[m][wk_stage],
            rb[n][wk_stage]);
      }
    }
  };


  auto store_c = [&](float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4])
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      #pragma unroll
      for (int n = 0; n < Cfg::acc_per_warp_n; n++)
      {
        int C_row = C_row_start + (m*Cfg::mma_m);
        int C_col = (C_col_start + (n*Cfg::mma_n))/2;
        float2 v0 = {rc[m][n][0], rc[m][n][1]}; 
        float2 v1 = {rc[m][n][2], rc[m][n][3]}; 
        C2[(C_row)*ldc2 + (C_col)] = v0; 
        C2[(C_row+8)*ldc2 + (C_col)] = v1;
      }
    }
  };

  for (int i = 0; i < Cfg::bk_stages; i++)
  {
    if (w == 0) TMA_load_A_B(i,i);
  }

  for (int i = 0; i < Cfg::warp_k_stages-1; i++)
  {
    mbarrier_wait_parity(smem.full(0),0);
    ldm_A_k(ra,i,0,i);
    ldm_B_k(rb,i,0,i);
  }
  
  static constexpr int num_full_iters = (Cfg::block_k_iters - Cfg::bk_stages)*Cfg::warp_k_iters;

  for (int j = 0; j < num_full_iters; j++)
  {
    int wk_compute_idx = j % Cfg::warp_k_iters;
    int wk_compute_stage = j % Cfg::warp_k_stages;
    int wk_load_idx = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
    int wk_load_stage = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_stages; 
    int bk_load_idx = ((j/Cfg::warp_k_iters) + Cfg::bk_stages);
    int bk_load_stage = ((j/Cfg::warp_k_iters) + Cfg::bk_stages) % Cfg::bk_stages;

    int bk_consume_idx = ((j + (Cfg::warp_k_stages-1))/Cfg::warp_k_iters);
    int bk_consume_stage = bk_consume_idx % Cfg::bk_stages; 
    int bk_consume_cycle = bk_consume_idx / Cfg:: bk_stages; 
    int wait_parity = bk_consume_cycle % 2;

    if (wk_load_idx == 0)
    { 
      mbarrier_wait_parity(smem.full(bk_consume_stage),wait_parity);
      if (w == 0) TMA_load_A_B(bk_load_idx,bk_load_stage);
      
    }
    ldm_A_k(ra,wk_load_idx,bk_consume_stage,wk_load_stage);
    ldm_B_k(rb,wk_load_idx,bk_consume_stage,wk_load_stage);
    mma_k(rc,ra,rb,wk_compute_idx,wk_compute_stage);

  }

  static constexpr int num_epilogue_iters = (Cfg::bk_stages-1)*Cfg::warp_k_iters;

  for (int j = num_full_iters; j < num_full_iters + num_epilogue_iters; j++)
  {

    int wk_compute_idx = j % Cfg::warp_k_iters;
    int wk_compute_stage = j % Cfg::warp_k_stages;
    int wk_load_idx = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
    int wk_load_stage = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_stages; 

    int bk_consume_idx = ((j + (Cfg::warp_k_stages-1))/Cfg::warp_k_iters);
    int bk_consume_stage = bk_consume_idx % Cfg::bk_stages; 
    int bk_consume_cycle = bk_consume_idx / Cfg:: bk_stages; 
    int wait_parity = bk_consume_cycle % 2;

    if (wk_load_idx == 0)
    {
      mbarrier_wait_parity(smem.full(bk_consume_stage),wait_parity);
    }
    ldm_A_k(ra,wk_load_idx,bk_consume_stage,wk_load_stage);
    ldm_B_k(rb,wk_load_idx,bk_consume_stage,wk_load_stage);
    mma_k(rc,ra,rb,wk_compute_idx,wk_compute_stage);
    
  }

  static constexpr int final_start = num_full_iters + num_epilogue_iters;
  static constexpr int drain_iters = Cfg::warp_k_iters - (Cfg::warp_k_stages-1);

  for (int j = final_start; j < final_start + drain_iters; j++)
  {
    __syncthreads();
    int wk_compute_idx = j % Cfg::warp_k_iters;
    int wk_compute_stage = j % Cfg::warp_k_stages;
    int wk_load_idx = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
    int wk_load_stage = (j + (Cfg::warp_k_stages-1)) % Cfg::warp_k_stages; 

    int bk_consume_idx = ((j + (Cfg::warp_k_stages-1))/Cfg::warp_k_iters);
    int bk_consume_stage = bk_consume_idx % Cfg::bk_stages; 

    ldm_A_k(ra,wk_load_idx,bk_consume_stage,wk_load_stage);
    ldm_B_k(rb,wk_load_idx,bk_consume_stage,wk_load_stage);
    mma_k(rc,ra,rb,wk_compute_idx,wk_compute_stage);
    
  }

  static constexpr int final_final_start = final_start + drain_iters;
  static constexpr int last_mma_iters = Cfg::warp_k_iters - (drain_iters);

  for (int j = final_final_start; j < final_final_start + last_mma_iters; j++)
  {
    int wk_compute_idx = j % Cfg::warp_k_iters;
    int wk_compute_stage = j % Cfg::warp_k_stages;
    mma_k(rc,ra,rb,wk_compute_idx,wk_compute_stage);
  }


  store_c(rc);
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev
)

{
  launcher.launch(matmul_kernel<Cfg>, a_map,b_map,C_dev);
}




__global__ void naive_gemm_ref(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++)
    {
        float a = __bfloat162float(A[row * K + k]);
        float b = __bfloat162float(B[col * K + k]); // col-major B
        acc += a * b;
    }

    C[row * N + col] = acc;
}

void verify_result(
    const NaiveTensor<float>& C,
    const NaiveTensor<float>& C_ref,
    int M, int N,
    float abs_thresh = 1e-1f,
    int max_print    = 20)
{
    int bad = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    const float eps = 1e-6f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float ref = C_ref.h_ptr[i * N + j];
            float gpu = C.h_ptr[i * N + j];

            float abs_err = fabsf(ref - gpu);
            float rel_err = abs_err / fmaxf(fabsf(ref), eps);

            max_abs = fmaxf(max_abs, abs_err);
            max_rel = fmaxf(max_rel, rel_err);

            if (abs_err > abs_thresh)
            {
                if (bad < max_print)
                {
                    printf("BAD (%d,%d) ref=%f gpu=%f abs=%e rel=%e\n",
                           i, j, ref, gpu, abs_err, rel_err);
                }
                bad++;
            }
        }
    }

    printf("\n==== Verification ====\n");
    printf("Max abs error : %e\n", max_abs);
    printf("Max rel error : %e\n", max_rel);
    printf("Bad elements  : %d\n", bad);

    if (bad == 0) printf("✅ PASSED\n");
    else          printf("❌ FAILED\n");
}

int main()
{
    using Cfg = GemmConfig;

    NaiveTensor<nv_bfloat16> A({Cfg::M, Cfg::K}, Layout::ROW_MAJOR);
    NaiveTensor<nv_bfloat16> B({Cfg::K, Cfg::N}, Layout::COL_MAJOR);

    NaiveTensor<float> C({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);
    NaiveTensor<float> C_ref({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);

    A.allocate();
    B.allocate();
    C.allocate();
    C_ref.allocate();

    A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

    A.to_device();
    B.to_device();
    C.to_device();
    C_ref.to_device();

    CUtensorMap a_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            A.d_ptr,
            {Cfg::M, Cfg::K},
            {Cfg::BM, Cfg::BK},
            Cfg::swizzle_mode
            
        );

    CUtensorMap b_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            B.d_ptr,
            {Cfg::K, Cfg::N},
            {Cfg::BK, Cfg::BN},
            Cfg::swizzle_mode
            
        );

    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    launcher.launch(matmul_kernel<Cfg>, a_map, b_map, C.d_ptr);
    cudaDeviceSynchronize();

    C.to_host();
    printf("Kernel launch complete.\n");

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

    verify_result(C, C_ref, Cfg::M, Cfg::N);
    

    return 0;

}
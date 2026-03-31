#pragma once
#include "../atoms/all.cuh"
#include "smem_allocator.cuh"
#include <math.h>
#include <intra_kernel_profiler/trace/trace.cuh>
namespace wa = warp_function;

namespace ikp = intra_kernel_profiler::trace;

enum ProfileRegion : uint16_t {
    kTotal         = 1,
    kLoadAB_fire   = 2,
    kLoadAB_wait   = 3,
    kLoadAB_smem   = 4,
    kMma            = 5,
    kStore_c       = 6,
};

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C,
    ikp::GlobalBuffer ikp_buf
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
  uint32_t b_ld_base = ((warp_start_n + ((l%8)+(8*(l/16))))*Cfg::BK + (8*((l/8)%2)))*sizeof(nv_bfloat16);
  float2* C2 = reinterpret_cast<float2*>(C);
  int ldc2 = Cfg::N/2;
  uint32_t ra[Cfg::warp_k_stages][Cfg::acc_per_warp_m][4];
  uint32_t rb[Cfg::warp_k_stages][Cfg::acc_per_warp_n][2];
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

  // IKP trace context init — only profile block 0
  const bool ikp_on = (b == 0);
  using TraceCtx = IKP_TRACE_CTX_TYPE(32768, Cfg::num_warps);
  TraceCtx ikp_ctx;
  if (ikp_on) { IKP_TRACE_CTX_INIT(ikp_ctx); }
  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kTotal, 0, ikp_on);

  auto TMA_load_A_B = [&](int bk_idx, int stage)
  {
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_fire, 0, ikp_on);
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
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_fire, 1, ikp_on);
  };

  auto ldm_A_k = [&](uint32_t ra[Cfg::warp_k_stages][Cfg::acc_per_warp_m][4], int k, int stage, int wk_stage)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      uint32_t a_ld_addr = smem.A(stage) + compact_swizzle<Cfg::swizzle_num>(a_ld_base + ((m*Cfg::mma_m*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x4_b16(ra[wk_stage][m], a_ld_addr);
    }

  };

  auto ldm_B_k = [&](uint32_t rb[Cfg::warp_k_stages][Cfg::acc_per_warp_n][2], int k, int stage, int wk_stage)
  {
    #pragma unroll 
    for (int n = 0; n < Cfg:: acc_per_warp_n/2; n++)
    {
      uint32_t b_ld_addr = smem.B(stage) + compact_swizzle<Cfg::swizzle_num>(b_ld_base + ((2*n*Cfg::mma_n*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x4_b16(rb[wk_stage][2*n], b_ld_addr);
    }

  };  

  auto mma_k = [&](float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4],
                  uint32_t ra[Cfg::warp_k_stages][Cfg::acc_per_warp_m][4],
                  uint32_t rb[Cfg::warp_k_stages][Cfg::acc_per_warp_n][2],
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
            ra[wk_stage][m],
            rb[wk_stage][n]);
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
  #pragma unroll
  for (int i = 0; i < Cfg::bk_stages; i++)
  {
    if (w == 0) TMA_load_A_B(i,i);
  }
  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 0, ikp_on);
  mbarrier_wait_parity(smem.full(0),0);
  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 1, ikp_on);

  #pragma unroll
  for (int i = 0; i < Cfg::warp_k_stages-1; i++)
  {
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
    ldm_A_k(ra,i,0,i);
    ldm_B_k(rb,i,0,i);
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
  }
  
  static constexpr int full_bk_iters = Cfg::block_k_iters - Cfg::bk_stages;
  static constexpr int wk_iters = Cfg::warp_k_iters - (Cfg::warp_k_stages - 1);

  for (int bk_idx = 0; bk_idx < full_bk_iters; bk_idx++)
  {
    int bk_cons_stage = bk_idx % Cfg::bk_stages; 
    int next_bk_cons_stage = (bk_idx + 1) % Cfg::bk_stages;
    int parity = ((bk_idx + 1) / Cfg::bk_stages) % 2;

    int next_bk_load_idx = (bk_idx + Cfg::bk_stages);
    int next_bk_load_stage = next_bk_load_idx % Cfg::bk_stages;
    int bk_base = bk_idx*Cfg::warp_k_iters;

    for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++)
    {
      int wk_load_idx = (wk_idx + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
      int wk_load_stage = (bk_base + wk_load_idx) % Cfg::warp_k_stages;
      int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
      ldm_A_k(ra,wk_load_idx,bk_cons_stage,wk_load_stage);
      ldm_B_k(rb,wk_load_idx,bk_cons_stage,wk_load_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
      mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
    }
    __syncthreads(); //wait for loads of curr bk stage to be done.
    if (w == 0) TMA_load_A_B(next_bk_load_idx,next_bk_load_stage);

    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 0, ikp_on);
    mbarrier_wait_parity(smem.full(next_bk_cons_stage),parity);
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 1, ikp_on);
    __syncthreads();
    
    for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++)
    {
      int wk_load_idx = (wk_idx + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
      int wk_load_stage = (bk_base + wk_load_idx) % Cfg::warp_k_stages;
      int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;

      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
      ldm_A_k(ra,wk_load_idx,bk_cons_stage,wk_load_stage);
      ldm_B_k(rb,wk_load_idx,bk_cons_stage,wk_load_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
      mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
    }

  }

  static constexpr int no_tma_end = full_bk_iters + (Cfg::bk_stages-1);

  for (int bk_idx = full_bk_iters; bk_idx < no_tma_end; bk_idx++)
  {
    int bk_cons_stage = bk_idx % Cfg::bk_stages; 
    int next_bk_cons_stage = (bk_idx + 1) % Cfg::bk_stages;
    int parity = ((bk_idx + 1) / Cfg::bk_stages) % 2;

    int bk_base = bk_idx*Cfg::warp_k_iters;

    for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++)
    {
      int wk_load_idx = (wk_idx + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
      int wk_load_stage = (bk_base + wk_load_idx) % Cfg::warp_k_stages;
      int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;

      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
      ldm_A_k(ra,wk_load_idx,bk_cons_stage,wk_load_stage);
      ldm_B_k(rb,wk_load_idx,bk_cons_stage,wk_load_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
      mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
    }
    __syncthreads(); //wait for loads of curr bk stage to be done.
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 0, ikp_on);
    mbarrier_wait_parity(smem.full(next_bk_cons_stage),parity);
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_wait, 1, ikp_on);
    __syncthreads();
    for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++)
    {
      int wk_load_idx = (wk_idx + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
      int wk_load_stage = (bk_base + wk_load_idx) % Cfg::warp_k_stages;
      int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;

      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
      ldm_A_k(ra,wk_load_idx,bk_cons_stage,wk_load_stage);
      ldm_B_k(rb,wk_load_idx,bk_cons_stage,wk_load_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
      mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
    }

  }

  static constexpr int bk_idx = Cfg::block_k_iters - 1; //epilogue 
  static constexpr int bk_cons_stage = bk_idx % Cfg::bk_stages; 
  static constexpr int bk_base = bk_idx*Cfg::warp_k_iters;

  for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++)
  {
    int wk_load_idx = (wk_idx + (Cfg::warp_k_stages-1)) % Cfg::warp_k_iters;
    int wk_load_stage = (bk_base + wk_load_idx) % Cfg::warp_k_stages;
    int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;

      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 0, ikp_on);
      ldm_A_k(ra,wk_load_idx,bk_cons_stage,wk_load_stage);
      ldm_B_k(rb,wk_load_idx,bk_cons_stage,wk_load_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kLoadAB_smem, 1, ikp_on);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
      mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
      IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
  }
  __syncthreads(); 

  for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++)
  {
    int wk_compute_stage = (bk_base + wk_idx) % Cfg::warp_k_stages;
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 0, ikp_on);
    mma_k(rc,ra,rb,wk_idx,wk_compute_stage);
    IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kMma, 1, ikp_on);
  }

  __syncthreads(); //syncthreads after store is ABSOLUTELY NEEDED
  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kStore_c, 0, ikp_on);
  store_c(rc);
  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kStore_c, 1, ikp_on);

  IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, kTotal, 1, ikp_on);
  if (ikp_on) { IKP_TRACE_CTX_FLUSH(ikp_ctx, ikp_buf); }
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev,
    ikp::GlobalBuffer ikp_buf
)

{
  launcher.launch(matmul_kernel<Cfg>, a_map,b_map,C_dev, ikp_buf);
}
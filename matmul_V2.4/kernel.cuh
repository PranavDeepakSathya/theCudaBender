#pragma once

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

  if (t == 0)
  {
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
    {
      mbarrier_init(smem.full(stage),1); 
      mbarrier_init(smem.empty(stage), Cfg::warps_per_block_m*Cfg::warps_per_block_n*32);
      
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
  };

  auto ldm_A_k = [&](uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4], int k, int stage)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      uint32_t a_ld_addr = smem.A(stage) + compact_swizzle<Cfg::swizzle_num>(a_ld_base + ((m*Cfg::mma_m*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x4_b16(ra[m][k], a_ld_addr);
    }

  };

  auto ldm_B_k = [&](uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2], int k, int stage)
  {
    #pragma unroll 
    for (int n = 0; n < Cfg:: acc_per_warp_n; n++)
    {
      uint32_t b_ld_addr = smem.B(stage) + compact_swizzle<Cfg::swizzle_num>(b_ld_base + ((n*Cfg::mma_n*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x2_b16(rb[n][k], b_ld_addr);
    }

  };  

  auto mma_k = [&](float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4],
                  uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4],
                  uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2],
                  int k)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      #pragma unroll
      for (int n = 0; n < Cfg::acc_per_warp_n; n++)
      {
        wa::mma_m16n8k16_row_col_f32_bf16(
            rc[m][n],
            ra[m][k],
            rb[n][k]);
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

  if (w == Cfg::producer_warp_id)
  {
    int stage = 0; 
    int phase = 1; 
    for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx++)
    {
      mbarrier_wait_parity(smem.empty(stage),phase);
      TMA_load_A_B(bk_idx,stage);
      stage = (stage + 1) % Cfg::bk_stages;
      if (stage == 0) phase ^= 1;
    }
  }
  else
  {
    int stage = 0; 
    int phase = 0; 


    uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4]; 
    uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
    float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0};
    
    mbarrier_wait_parity(smem.full(stage),phase);
    //outer_prologue
    ldm_A_k(ra,0,stage);
    ldm_B_k(rb,0,stage);

    for (int bk_idx = 0; bk_idx < Cfg::block_k_iters-1; bk_idx++)
    {
      int next_stage = stage + 1;
      int next_phase = phase;

      if (next_stage == Cfg::bk_stages) {
          next_stage = 0;
          next_phase ^= 1;
      }

      for (int k = 0; k < Cfg::warp_k_iters-1;k++)
      {
        ldm_A_k(ra,k+1,stage);
        ldm_B_k(rb,k+1,stage);
        mma_k(rc,ra,rb,k);
      }
      mbarrier_arrive(smem.empty(stage));
      mma_k(rc,ra,rb,Cfg::warp_k_iters-1);
      mbarrier_wait_parity(smem.full(next_stage),next_phase);
      ldm_A_k(ra,0,next_stage);
      ldm_B_k(rb,0,next_stage);

      stage = next_stage;
      phase = next_phase;

    }
    #pragma unroll
    for (int k = 0; k < Cfg::warp_k_iters - 1; k++)
    {
      ldm_A_k(ra, k+1, stage);
      ldm_B_k(rb, k+1, stage);
      mma_k(rc, ra, rb, k);
    }

    mma_k(rc, ra, rb, Cfg::warp_k_iters - 1);
    mbarrier_arrive(smem.empty(stage));

    store_c(rc);
  }

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
#pragma once

#include "../atoms/all.cuh"
#include "config.cuh"

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
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t As = smem_addr; 
  uint32_t Bs = smem_addr + (Cfg::As_bytes*Cfg::bk_stages);
  uint32_t full_addr = smem_addr + (Cfg::bk_stages*(Cfg::As_bytes+Cfg::Bs_bytes)); 
  uint32_t empty_addr = full_addr + (Cfg::bk_stages*8); 
  int b = blockIdx.x;
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t % 32; 
  int block_start_m, block_start_n; 
  tile_sched::block_swizzle<Cfg::group_m, Cfg::group_n, Cfg::blocks_per_group, Cfg::G_outer_M, Cfg::G_outer_N, Cfg::BM,Cfg::BN>(b,block_start_m,block_start_n);
  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN;

  if (t == 0)
  {
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
    {
      mbarrier_init(full_addr + (stage*8),1); 
      mbarrier_init(empty_addr + (stage*8), Cfg::warps_per_block_m*Cfg::warps_per_block_n*32);
      
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  auto TMA_load_A_B = [&](int bk_idx, int stage)
    
  {
    uint32_t As_stage = As + (Cfg::As_bytes*stage);
    uint32_t Bs_stage = Bs + (Cfg::Bs_bytes*stage);
    int k_offset = bk_idx*Cfg::BK; 
    uint32_t curr_full_addr = full_addr + (stage*8);
    cp_async_bulk_tensor_2d(As_stage, &a_map, k_offset, block_start_m, curr_full_addr);
    cp_async_bulk_tensor_2d(Bs_stage, &b_map, k_offset, block_start_n, curr_full_addr);
    mbarrier_arrive_expect_tx(curr_full_addr, Cfg::As_bytes + Cfg::Bs_bytes);
  };

  uint32_t a_ld_base = ((warp_start_m + (l%16))*Cfg::BK + (8*(l/16)))*sizeof(nv_bfloat16);
  uint32_t b_ld_base = ((warp_start_n + (l%8))*Cfg::BK + (8*(l/8)))*sizeof(nv_bfloat16);

  auto Consume_A_B = [&](int stage,
    uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4],
    uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2],
    float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4])
  {
    uint32_t curr_empty_addr = empty_addr + (stage*8);
    #pragma unroll
    for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++){
       #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
      {
        uint32_t a_ld_offset = compact_swizzle<Cfg::swizzle_num>(a_ld_base + ((wm_idx*Cfg::mma_m)*Cfg::BK + (wk_idx*Cfg::mma_k))*sizeof(nv_bfloat16));
        uint32_t a_ld_addr = (As + (stage*Cfg::As_bytes)) + a_ld_offset;
        wa::ldmatrix_m8n8_x4_b16(ra[wm_idx][wk_idx], a_ld_addr);
      }
    }
    #pragma unroll
    for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++){
      #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters/2; wk_idx++)
      {
        uint32_t b_ld_offset = compact_swizzle<Cfg::swizzle_num>(b_ld_base + ((wn_idx*Cfg::mma_n)*Cfg::BK + (2*wk_idx*Cfg::mma_k))*sizeof(nv_bfloat16));
        uint32_t b_ld_addr = (Bs + (stage*Cfg::Bs_bytes)) + b_ld_offset;
        wa::ldmatrix_m8n8_x4_b16(rb[wn_idx][2*wk_idx], b_ld_addr);
      }
    }

    mbarrier_arrive(curr_empty_addr);
    #pragma unroll
    for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
    {
       #pragma unroll
      for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++)
      {
        #pragma unroll
        for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++)
          {
            wa::mma_m16n8k16_row_col_f32_bf16(rc[wm_idx][wn_idx], ra[wm_idx][wk_idx], rb[wn_idx][wk_idx]);
          }
      }
    }

  };

  if (w == Cfg::producer_warp_id)
  {
    if (l == 0)
    {
      int stage = 0;
      int phase = 1;
       #pragma unroll 
      for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx ++)
      {
        uint32_t curr_empty_addr = empty_addr + (stage*8);
        mbarrier_wait_parity(curr_empty_addr,phase);
        TMA_load_A_B(bk_idx,stage); 
        stage = (stage + 1) % Cfg::bk_stages;
        if (stage == 0)
          phase ^= 1;
      }
    }
  }
  else
  {
    int stage = 0;
    int phase = 0; 

    uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4];
    uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
    float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0}; 
    #pragma unroll
    for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx ++)
    {
      
      uint32_t curr_full_addr = full_addr + (stage*8); 
      if (w == 0)
        mbarrier_wait_parity(curr_full_addr,phase);
      asm volatile("bar.sync %0, %1;" :: "n"(1), "n"(Cfg::warps_per_block_m*Cfg::warps_per_block_n*32));
      //tma_fence(); 

      Consume_A_B(stage,ra,rb,rc); 

      stage = (stage + 1) % Cfg::bk_stages;
      if (stage == 0)
      phase ^= 1;
    }

    float2* C2 = reinterpret_cast<float2*>(C); 
    int lane_row = l/4; 
    int lane_col = 2*(l%4); 
    int ldc2 = Cfg::N/2;

    #pragma unroll
    for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++)
    {
      #pragma unroll
      for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++)
      {
        int C_row = block_start_m + warp_start_m + (wm_idx*Cfg::mma_m) + lane_row;
        int C_col = (block_start_n + warp_start_n + (wn_idx*Cfg::mma_n) + lane_col)/2;
  
        float2 v0 = {rc[wm_idx][wn_idx][0], rc[wm_idx][wn_idx][1]}; 
        float2 v1 = {rc[wm_idx][wn_idx][2], rc[wm_idx][wn_idx][3]}; 
        C2[(C_row)*ldc2 + (C_col)] = v0; 
        C2[(C_row+8)*ldc2 + (C_col)] = v1;
      }
    }
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
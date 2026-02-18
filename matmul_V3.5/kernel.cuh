#pragma once

#include "../atoms/all.cuh"
#include "config.cuh"

namespace ptx = cuda::ptx;
namespace wa = warp_function;

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    __grid_constant__ const CUtensorMap c_map
)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t As = smem_addr; 
  uint32_t Bs = smem_addr + (Cfg::As_bytes*Cfg::bk_stages);
  uint32_t Cs = smem_addr + (Cfg::bk_stages*(Cfg::As_bytes+Cfg::Bs_bytes));
  float2* Cs_ptr = reinterpret_cast<float2*>(__cvta_shared_to_generic(Cs));

  uint32_t full_addr = smem_addr + (Cfg::Cs_bytes + Cfg::bk_stages*(Cfg::As_bytes+Cfg::Bs_bytes)); 
  uint32_t empty_addr = full_addr + (Cfg::bk_stages*8); 

  int b = blockIdx.x;
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t % 32; 

  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN;
  int C_row_start =  warp_start_m + (l/4);
  int C_col_start =  warp_start_n + 2*(l%4);
  int ldc2 = Cfg::BN/2;
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


  auto store_Cs = [&](float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4])
  {
    #pragma unroll
    for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++)
    {
      #pragma unroll
      for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++)
      {
        int C_row = C_row_start + (wm_idx*Cfg::mma_m);
        int C_col = (C_col_start + (wn_idx*Cfg::mma_n))/2;
  
        float2 v0 = {rc[wm_idx][wn_idx][0], rc[wm_idx][wn_idx][1]}; 
        float2 v1 = {rc[wm_idx][wn_idx][2], rc[wm_idx][wn_idx][3]}; 
        Cs_ptr[(C_row)*ldc2 + (C_col)] = v0; 
        Cs_ptr[(C_row+8)*ldc2 + (C_col)] = v1;
      }
    }
  };

  auto TMA_store_C = [&](int block_start_m, int block_start_n)
  {
    cp_async_bulk_tensor_2d_store(&c_map, block_start_m, block_start_n, Cs);
    cp_async_commit_group();
  };

  auto TMA_load_A_B = [&](int bk_idx, int stage, int block_start_m, int block_start_n)
    
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
      int phase = 0;
      for (int iter = 0; iter < Cfg::persist_num_iters; iter++)
      {
        int tile_id = tile_sched::get_linear_tile_id<Cfg::num_tiles,Cfg::num_sms>(iter,b);
        if(tile_id == -1) break;
        uint32_t m, n;
        tile_sched::morton_decode_2d((uint32_t)tile_id, m, n);
        int block_start_m = m*Cfg::BM;
        int block_start_n = n*Cfg::BN;
        
        for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; ++bk_idx, ++stage)
        {
          if (stage == Cfg::bk_stages)
          {
            stage = 0;
            phase ^=1;
          }
          uint32_t curr_empty_addr = empty_addr + (stage*8);
          mbarrier_wait_parity(curr_empty_addr,phase);
          TMA_load_A_B(bk_idx,stage, block_start_m, block_start_n);

        }
      }
    }
  }
  else
  {
    int stage = 0;
    int phase = 0;
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
    {
      mbarrier_arrive(empty_addr + (8*stage));
    }
    
    for (int iter = 0; iter < Cfg::persist_num_iters; iter++)
    {
      int tile_id = tile_sched::get_linear_tile_id<Cfg::num_tiles,Cfg::num_sms>(iter,b);
      if(tile_id == -1) break;
      uint32_t m, n;
      tile_sched::morton_decode_2d((uint32_t)tile_id, m, n);
      int block_start_m = m*Cfg::BM;
      int block_start_n = n*Cfg::BN;
      uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4];
      uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
      float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0}; 

      for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; ++bk_idx, ++stage)
      {
        if (stage == Cfg::bk_stages)
        {
          stage = 0;
          phase ^=1;
        }
        uint32_t curr_full_addr = full_addr + (stage*8); 
        if (w == 0)
          mbarrier_wait_parity(curr_full_addr,phase);
        asm volatile("bar.sync %0, %1;" :: "n"(1), "n"(Cfg::warps_per_block_m*Cfg::warps_per_block_n*32));
        Consume_A_B(stage,ra,rb,rc); 

      }

      cp_async_wait_group<0>(); //wait for prev tma store to be finished before writing to smem 

      store_Cs(rc);
      asm volatile("bar.sync %0, %1;" :: "n"(1), "n"(Cfg::warps_per_block_m*Cfg::warps_per_block_n*32));
      tma_fence();
      if (t == 0) TMA_store_C(block_start_m, block_start_n);
    }
  }
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    CUtensorMap c_map
)

{
  launcher.launch(matmul_kernel<Cfg>, a_map,b_map,c_map);
}
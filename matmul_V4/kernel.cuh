// kernel.cuh
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
  uint8_t* ptr = smem_raw;
  nv_bfloat16* As[Cfg::bk_stages];
  nv_bfloat16* Bs[Cfg::bk_stages]; 
  uint32_t As_base[Cfg::bk_stages];
  uint32_t Bs_base[Cfg::bk_stages]; 


  #pragma unroll 
  for (int stage = 0; stage < Cfg::bk_stages; stage++)
  {
    As[stage] = alloc<nv_bfloat16,1024>(ptr, Cfg::BM*Cfg::BK); 
    As_base[stage]  = static_cast<uint32_t>(__cvta_generic_to_shared(As[stage]));
  }

  #pragma unroll 
  for (int stage = 0; stage < Cfg::bk_stages; stage++)
  {
    Bs[stage] = alloc<nv_bfloat16,1024>(ptr, Cfg::BK*Cfg::BN); 
    Bs_base[stage] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[stage]));
  }

  uint64_t* bar = alloc<uint64_t,8>(ptr, Cfg::bk_stages);

  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int b = blockIdx.x; 

  int block_start_m = (b/Cfg::GN)*Cfg::BM;
  int block_start_n = (b%Cfg::GN)*Cfg::BN; 
  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN;

  uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4];
  uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
  float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0}; 


  if (t == 0)
  {
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
      ptx::mbarrier_init(&bar[stage],1);
    asm volatile("fence.mbarrier_init.release.cluster;");  
  }
  __syncthreads(); 


  auto tma_load_A_B = [&](int bk_idx, int stage) {
    if (t == 0)
    {
      int32_t A_coords[2] = {bk_idx*Cfg::BK, block_start_m};
      int32_t B_coords[2] = {bk_idx*Cfg::BK, block_start_n};
      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        As[stage], &a_map, A_coords, &bar[stage]);

      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        Bs[stage], &b_map, B_coords, &bar[stage]);

        uint64_t _ = ptx::mbarrier_arrive_expect_tx(
        ptx::sem_release, 
        ptx::scope_cta, 
        ptx::space_shared,
        &bar[stage],
        Cfg::As_bytes + Cfg::Bs_bytes
      );
    }
    
  };

  auto consume_on_bk_stage = [&](int stage)
  {
    #pragma unroll
    for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++)
    {
      #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
      {
        int a_ld_shared_offset = (warp_start_m + (wm_idx*Cfg::mma_m) + (l%16))*Cfg::BK + (wk_idx*Cfg::mma_k + (8*(l/16)));
        uint32_t a_ld_addr = As_base[stage] + (a_ld_shared_offset*sizeof(nv_bfloat16));
        wa::ldmatrix_m8n8_x4_b16(ra[wm_idx][wk_idx], a_ld_addr);

      }
    }

    #pragma unroll
    for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++)
    {
      #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters/2; wk_idx++) //load 2 warp_k iters at once
      {
        int b_ld_shared_offset = (warp_start_n + (wn_idx*Cfg::mma_n) + (l%8))*Cfg::BK + ((2*wk_idx*Cfg::mma_k) + (8*(l/8)));
        uint32_t b_ld_addr = Bs_base[stage] + (b_ld_shared_offset*sizeof(nv_bfloat16));
        wa::ldmatrix_m8n8_x4_b16(rb[wn_idx][2*wk_idx], b_ld_addr);

      }
    }
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

  for (int stage = 0; stage < Cfg::bk_stages - 1; stage++) //prologue
    tma_load_A_B(stage, stage);

  int stage = 0;
  int phase = 0;
  for (int bk_idx = 0; bk_idx < Cfg::block_k_iters - (Cfg::bk_stages-1); bk_idx++)
  {
    __syncthreads();
    const int prefetch_bk_idx = bk_idx + (Cfg::bk_stages-1);
    tma_load_A_B(prefetch_bk_idx, prefetch_bk_idx % Cfg::bk_stages);
    if (w == 0)
    {
      while(!ptx::mbarrier_try_wait_parity(&bar[stage], phase)); 
    }
    __syncthreads();

    consume_on_bk_stage(stage);
    stage = (stage + 1) % Cfg::bk_stages;
    if (stage == 0)
      phase ^= 1;
  }

  
  for (int bk_idx = Cfg::block_k_iters - (Cfg::bk_stages-1); bk_idx < Cfg::block_k_iters; bk_idx++)
  {
    if (w == 0)
    {
      while(!ptx::mbarrier_try_wait_parity(&bar[stage], phase)); 
    }
    __syncthreads();

    consume_on_bk_stage(stage);
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
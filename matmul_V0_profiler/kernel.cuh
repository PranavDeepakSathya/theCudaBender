// kernel.cuh
#pragma once

#include "../atoms/all.cuh"
#include "config.cuh"
#include "profile.cuh"

namespace ptx = cuda::ptx;
namespace wa = warp_function;


template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C,
    int64_t* profiler,
    int num_prof_entries
)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  constexpr int WARPS_PER_BLOCK = Cfg::warps_per_block_m * Cfg::warps_per_block_n;
  uint8_t* ptr = smem_raw;
  nv_bfloat16* As = alloc<nv_bfloat16,1024>(ptr, Cfg::BM*Cfg::BK);
  uint32_t As_base = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  nv_bfloat16* Bs = alloc<nv_bfloat16,1024>(ptr,Cfg::BK*Cfg::BN); 
  uint32_t Bs_base = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));
  uint64_t* bar = alloc<uint64_t, 8>(ptr, 1);
  uint64_t bar_token; 

  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int b = blockIdx.x; 

  int warp_stream = b * WARPS_PER_BLOCK + w;
  Profiler prof;

  if (is_elected())
    prof.init(num_prof_entries, profiler, warp_stream);

  if (is_elected()) prof.start(TAG_SETUP);
  
  int block_start_m = (b/Cfg::GN)*Cfg::BM;
  int block_start_n = (b%Cfg::GN)*Cfg::BN; 
  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN;

  uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4];
  uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
  float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0}; 


  if (t == 0)
  {
    ptx::mbarrier_init(bar,Cfg::block_size);
    asm volatile("fence.mbarrier_init.release.cluster;");  
  }
  __syncthreads(); 

  if (is_elected()) prof.stop();
  
  for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx++)
  {
    int32_t A_coords[2] = {bk_idx*Cfg::BK, block_start_m};
    int32_t B_coords[2] = {bk_idx*Cfg::BK, block_start_n}; 
    __syncthreads(); //wait for prev ldmatrix and mma stage to finish
   
    if (w == 0 && is_elected()) prof.start(TAG_TMA);
    if (t == 0)
    {
      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        As, &a_map, A_coords, bar);

      ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
        Bs, &b_map, B_coords, bar);

        bar_token = ptx::mbarrier_arrive_expect_tx(
        ptx::sem_release, 
        ptx::scope_cta, 
        ptx::space_shared,
        bar,
        Cfg::As_bytes + Cfg::Bs_bytes
      );
    }
    else
    {
      bar_token = ptx::mbarrier_arrive(bar);
    }
    
    while(!ptx::mbarrier_try_wait(bar,bar_token));
    if(w == 0 && is_elected()) prof.stop();

    #pragma unroll
    for (int wm_idx = 0; wm_idx < Cfg::acc_per_warp_m; wm_idx++)
    {
      #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
      {
        int a_ld_shared_offset = (warp_start_m + (wm_idx*Cfg::mma_m) + (l%16))*Cfg::BK + (wk_idx*Cfg::mma_k + (8*(l/16)));
        uint32_t a_ld_addr = As_base + (a_ld_shared_offset*sizeof(nv_bfloat16));
        if(is_elected()) prof.start(TAG_LDMATRIX);
        wa::ldmatrix_m8n8_x4_b16(ra[wm_idx][wk_idx], a_ld_addr);
        if(is_elected()) prof.stop();

      }
    }

    #pragma unroll
    for (int wn_idx = 0; wn_idx < Cfg::acc_per_warp_n; wn_idx++)
    {
      #pragma unroll
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters/2; wk_idx++) //load 2 warp_k iters at once
      {
        int b_ld_shared_offset = (warp_start_n + (wn_idx*Cfg::mma_n) + (l%8))*Cfg::BK + ((2*wk_idx*Cfg::mma_k) + (8*(l/8)));
        uint32_t b_ld_addr = Bs_base + (b_ld_shared_offset*sizeof(nv_bfloat16));
        if(is_elected()) prof.start(TAG_LDMATRIX);
        wa::ldmatrix_m8n8_x4_b16(rb[wn_idx][2*wk_idx], b_ld_addr);
        if(is_elected()) prof.stop();
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
          if(is_elected()) prof.start(TAG_MMA);
          wa::mma_m16n8k16_row_col_f32_bf16(rc[wm_idx][wn_idx], ra[wm_idx][wk_idx], rb[wn_idx][wk_idx]);
          if(is_elected()) prof.stop();
        }
      }
    }    

  }

  float2* C2 = reinterpret_cast<float2*>(C); 
  int lane_row = l/4; 
  int lane_col = 2*(l%4); 
  int ldc2 = Cfg::N/2;
  __syncthreads();
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
      if(is_elected()) prof.start(TAG_STORE);
      C2[(C_row)*ldc2 + (C_col)] = v0; 
      C2[(C_row+8)*ldc2 + (C_col)] = v1;
      if(is_elected()) prof.stop();
    }
  }
  if (is_elected()) prof.flush();
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev,
    int64_t* profiler,
    int num_prof_entries
)

{
  launcher.launch(matmul_kernel<Cfg>, a_map,b_map,C_dev, profiler, num_prof_entries);
}
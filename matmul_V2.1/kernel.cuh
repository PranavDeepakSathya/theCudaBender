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
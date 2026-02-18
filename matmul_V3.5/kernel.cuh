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
  uint32_t Cs = smem_addr + (Cfg::bk_stages*(Cfg::As_bytes+Cfg::Bs_bytes))
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
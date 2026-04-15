#pragma once 
#include "../gemm_maker.cuh"

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C
)
{
  GemmMaker<Cfg> gemm(a_map, b_map, threadIdx.x);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  int b = blockIdx.x; 
  int t = threadIdx.x; 
  int l = t % 32;
  int w = t / 32; 
  int block_start_m; 
  int block_start_n; 
  block_swizzle<Cfg::group_m,Cfg::group_n,Cfg::blocks_per_group,Cfg::G_outer_M, Cfg::G_outer_N, Cfg::BM, Cfg::BN>(b,block_start_m,block_start_n);
  float2 *C2 = reinterpret_cast<float2*>(C); 

  uint32_t As = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t Bs = As + Cfg::As_bytes;
  uint32_t m_bar = Bs + Cfg::Bs_bytes; 

  uint32_t ra[Cfg::acc_per_warp_m][4];
  uint32_t rb[Cfg::acc_per_warp_n][2];
  float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0};

  if (t == 0)
  {
    mbarrier_init(m_bar,32);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  __syncthreads();

  int parity = 0; 
  for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx ++)
  {
    __syncthreads();
    if (w == 0)
    { 
      if (l == 0)
      {
        mbarrier_arrive_expect_tx(m_bar, Cfg::As_bytes + Cfg::Bs_bytes); 
        gemm.load_A_g2s(bk_idx, block_start_m, As, m_bar); 
        gemm.load_B_g2s(bk_idx, block_start_n, Bs, m_bar);
      }
      else
      {
        mbarrier_arrive(m_bar);
      }
    }
    __syncthreads(); 
    mbarrier_wait_parity(m_bar, parity); 

    for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
    {
      gemm.load_A_s2r(ra,As,wk_idx);
      gemm.load_B_s2r(rb,Bs,wk_idx); 
      gemm.mma(rc,ra,rb);
    }
    parity^=1;
  }
  __syncthreads();
  gemm.store_C(C2,rc,block_start_m,block_start_n);
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
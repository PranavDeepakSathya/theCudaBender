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

  uint32_t As_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t Bs_base = As_base + (Cfg::As_bytes*Cfg::bk_stages);
  uint32_t empty_bar_base = Bs_base + (Cfg::Bs_bytes*Cfg::bk_stages);
  uint32_t full_bar_base = empty_bar_base + (8*Cfg::bk_stages); 

  auto As        = [&](int s) { return As_base        + s * Cfg::As_bytes; };
  auto Bs        = [&](int s) { return Bs_base        + s * Cfg::Bs_bytes; };
  auto empty_bar = [&](int s) { return empty_bar_base + s * 8; };
  auto full_bar  = [&](int s) { return full_bar_base  + s * 8; };


  if (t == 0)
  {
    for (int s = 0; s < Cfg::bk_stages; s++)
    {
      mbarrier_init(empty_bar(s),Cfg::warps_per_block_m*Cfg::warps_per_block_n*32);
      mbarrier_init(full_bar(s),32);
    }
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  __syncthreads();

  
  if (w == Cfg::dma_warp_id)
  {
    int producer_parity = 1; 
    int stage = 0; 
    for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx++)
    {
      mbarrier_wait_parity(empty_bar(stage),producer_parity);
      if(l == 0)
      {
        mbarrier_arrive_expect_tx(full_bar(stage),Cfg::As_bytes + Cfg::Bs_bytes);
        gemm.load_A_g2s(bk_idx, block_start_m, As(stage), full_bar(stage));
        gemm.load_B_g2s(bk_idx, block_start_n, Bs(stage), full_bar(stage)); 
      }
      else
      {
        mbarrier_arrive(full_bar(stage));
      }
      stage = (stage + 1) % Cfg::bk_stages; 
      if (stage == 0) producer_parity ^= 1;
    }
  }
  else 
  {
    int consumer_parity = 0; 
    int stage = 0; 
    uint32_t ra[Cfg::acc_per_warp_m][4];
    uint32_t rb[Cfg::acc_per_warp_n][2];
    float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0};
    for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; bk_idx++)
    {
      mbarrier_wait_parity(full_bar(stage), consumer_parity); 
      for (int wk_idx = 0; wk_idx < Cfg::warp_k_iters; wk_idx++)
      {
        gemm.load_A_s2r(ra,As(stage),wk_idx);
        gemm.load_B_s2r(rb,Bs(stage),wk_idx); 
        gemm.mma(rc,ra,rb);
      }
      mbarrier_arrive(empty_bar(stage));
      stage = (stage + 1) % Cfg::bk_stages; 
      if (stage == 0) consumer_parity ^= 1;
    }
    sync_bar<Cfg::warps_per_block_m*Cfg::warps_per_block_n*32>(); 
    gemm.store_C(C2, rc, block_start_m,block_start_n);
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
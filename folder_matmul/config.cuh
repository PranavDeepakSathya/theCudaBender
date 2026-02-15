#pragma once
#include <cuda_bf16.h>
#include <cstdint>

struct GemmConfig
{
  static constexpr int M = 4096;
  static constexpr int N = 4096;
  static constexpr int K = 4096;

  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  static constexpr int acc_per_warp_m = 4;
  static constexpr int acc_per_warp_n = 4;
  static constexpr int warp_k_iters   = 4;

  static constexpr int warps_per_block_m = 2;
  static constexpr int warps_per_block_n = 4;

  static constexpr int WM = mma_m * acc_per_warp_m;
  static constexpr int WN = mma_n * acc_per_warp_n;

  static constexpr int BK = mma_k * warp_k_iters;

  static constexpr int BM = WM * warps_per_block_m;
  static constexpr int BN = WN * warps_per_block_n;

  static constexpr int num_warps     = warps_per_block_m * warps_per_block_n;
  static constexpr int block_size = num_warps * 32;
  static constexpr int block_k_iters = K/BK;

  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(nv_bfloat16);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(nv_bfloat16);

  static constexpr uint32_t smem_overhead = 4 * 1024;

  static constexpr uint32_t shared_bytes =
      As_bytes + Bs_bytes + smem_overhead;

  static constexpr int GM = M / BM;
  static constexpr int GN = N / BN;
  static constexpr int grid_size = GM * GN;

};
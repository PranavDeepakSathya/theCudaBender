#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#ifndef ACC_PER_WARP_M
#define ACC_PER_WARP_M 4
#endif

#ifndef ACC_PER_WARP_N
#define ACC_PER_WARP_N 4
#endif

#ifndef WARP_K_ITERS
#define WARP_K_ITERS 4
#endif

#ifndef WARPS_PER_BLOCK_M
#define WARPS_PER_BLOCK_M 2
#endif

#ifndef WARPS_PER_BLOCK_N
#define WARPS_PER_BLOCK_N 4
#endif

#ifndef BK_STAGES
#define BK_STAGES 2
#endif


static constexpr bool is_pow2(int x)
{
return x > 1 && ((x & (x - 1)) == 0);
}

struct GemmConfig
{
  static constexpr int M = 4096;
  static constexpr int N = 4096;
  static constexpr int K = 4096;

  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  static constexpr int acc_per_warp_m = ACC_PER_WARP_M;
  static constexpr int acc_per_warp_n = ACC_PER_WARP_N;
  static constexpr int warp_k_iters   = WARP_K_ITERS;

  static constexpr int warps_per_block_m = WARPS_PER_BLOCK_M;
  static constexpr int warps_per_block_n = WARPS_PER_BLOCK_N;

  static_assert(acc_per_warp_m > 1);
  static_assert(acc_per_warp_n > 1);
  static_assert(warp_k_iters   > 1);
  static_assert(warps_per_block_m > 1);
  static_assert(warps_per_block_n > 1);

  static_assert(is_pow2(acc_per_warp_m));
  static_assert(is_pow2(acc_per_warp_n));
  static_assert(is_pow2(warp_k_iters));
  static_assert(is_pow2(warps_per_block_m));
  static_assert(is_pow2(warps_per_block_n));

  static constexpr int WM = mma_m * acc_per_warp_m;
  static constexpr int WN = mma_n * acc_per_warp_n;
  static constexpr int BK = mma_k * warp_k_iters;

  static constexpr int BM = WM * warps_per_block_m;
  static constexpr int BN = WN * warps_per_block_n;
  static constexpr int bk_stages = BK_STAGES;

  static_assert(K % BK == 0);

  static constexpr int block_k_iters = K / BK;

  static constexpr int num_warps  = (warps_per_block_m * warps_per_block_n) + 1;
  static constexpr int producer_warp_id = warps_per_block_m*warps_per_block_n;
  static constexpr int block_size = num_warps * 32;

  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(nv_bfloat16);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(nv_bfloat16);

  static constexpr uint32_t smem_overhead = 4 * 1024;

  static constexpr uint32_t shared_bytes =
      ((As_bytes + Bs_bytes)*bk_stages) + smem_overhead;

  static_assert(shared_bytes <= 100 * 1024,
                "Config rejected: shared memory exceeds 100KB");


  static_assert(block_size < 1024);
  static_assert(M % BM == 0);
  static_assert(N % BN == 0);

  static constexpr int GM = M / BM;
  static constexpr int GN = N / BN;

  static constexpr int grid_size = GM * GN;
  
};


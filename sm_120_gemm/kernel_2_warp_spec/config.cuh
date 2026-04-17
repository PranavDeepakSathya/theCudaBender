#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#ifndef ACC_PER_WARP_M
#define ACC_PER_WARP_M 2
#endif
#ifndef ACC_PER_WARP_N
#define ACC_PER_WARP_N 2
#endif
#ifndef WARPS_PER_BLOCK_M
#define WARPS_PER_BLOCK_M 2
#endif
#ifndef WARPS_PER_BLOCK_N
#define WARPS_PER_BLOCK_N 4
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif
#ifndef GROUP_M 
#define GROUP_M 4
#endif
#ifndef GROUP_N 
#define GROUP_N 4
#endif
#ifndef BK_STAGES 
#define BK_STAGES 2
#endif


static constexpr bool is_pow2(int x) { return x > 1 && ((x & (x-1)) == 0); }

struct GemmConfig
{
  // ── problem size ────────────────────────────────────────────────────────────
  static constexpr int M = 8192, N = 8192, K = 8192;

  // ── mma shape ───────────────────────────────────────────────────────────────
  static constexpr int mma_m = 16, mma_n = 8, mma_k = 16;

  // ── tuning knobs ────────────────────────────────────────────────────────────
  static constexpr int acc_per_warp_m    = ACC_PER_WARP_M;
  static constexpr int acc_per_warp_n    = ACC_PER_WARP_N;
  static constexpr int warps_per_block_m = WARPS_PER_BLOCK_M;
  static constexpr int warps_per_block_n = WARPS_PER_BLOCK_N;
  static constexpr int BK                = BLOCK_K;

  static_assert(acc_per_warp_m > 1 && is_pow2(acc_per_warp_m));
  static_assert(acc_per_warp_n > 1 && is_pow2(acc_per_warp_n));
  static_assert(warps_per_block_m > 1 && is_pow2(warps_per_block_m));
  static_assert(warps_per_block_n > 1 && is_pow2(warps_per_block_n));
  static_assert(is_pow2(BK) && BK % mma_k == 0);

  // ── derived tile sizes ──────────────────────────────────────────────────────
  static constexpr int WM = mma_m * acc_per_warp_m;
  static constexpr int WN = mma_n * acc_per_warp_n;
  static constexpr int BM = WM * warps_per_block_m;
  static constexpr int BN = WN * warps_per_block_n;

  static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0);

  // ── iteration counts ────────────────────────────────────────────────────────
  static constexpr int warp_k_iters  = BK / mma_k;
  static constexpr int block_k_iters = K  / BK;
  static constexpr int bk_stages = BK_STAGES;


  // ── block / grid ────────────────────────────────────────────────────────────
  static constexpr int dma_warp_id  = warps_per_block_m * warps_per_block_n;
  static constexpr int num_warps = dma_warp_id + 1; 
  static constexpr int block_size = num_warps * 32;

  static_assert(block_size < 1024);

  static_assert(M % BM == 0);
  static_assert(N % BN == 0);

  static constexpr int GM = M / BM;
  static constexpr int GN = N / BN;
  static constexpr int grid_size = GM * GN;
  static constexpr int group_m = GROUP_M;
  static constexpr int group_n = GROUP_N;
  static_assert(GM % group_m == 0);
  static_assert(GN % group_n == 0);
  static_assert(group_m <= GM);
  static_assert(group_n <= GN);

  static constexpr int blocks_per_group = group_m * group_n;

  static constexpr int G_outer_M = GM / group_m;
  static constexpr int G_outer_N = GN / group_n;

  static constexpr uint32_t ld_bytes =
      BK * sizeof(nv_bfloat16);

  static constexpr CUtensorMapSwizzle swizzle_mode =
      (ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                          CU_TENSOR_MAP_SWIZZLE_NONE;


  static constexpr int swizzle_num =
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 128  :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 384  :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 896 :
                                                    0;
  



  // ── smem footprint ──────────────────────────────────────────────────────────
  static constexpr uint32_t As_bytes       = BM * BK * sizeof(nv_bfloat16);
  static constexpr uint32_t Bs_bytes       = BK * BN * sizeof(nv_bfloat16);
  static constexpr uint32_t barrier_bytes  = 16;
  static constexpr uint32_t shared_bytes   = bk_stages*(As_bytes + Bs_bytes + barrier_bytes);
  
  
};
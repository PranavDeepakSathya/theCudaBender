#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
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

#ifndef C_STAGES
#define C_STAGES 2
#endif

#ifndef NUM_SMS
#define NUM_SMS 188
#endif

static constexpr bool is_pow2(int x)
{
  return x > 1 && ((x & (x - 1)) == 0);
}

struct GemmConfig
{
  // ============================================================
  // Problem Size
  // ============================================================

  static constexpr int M = 8192;
  static constexpr int N = 8192;
  static constexpr int K = 8192;

  // ============================================================
  // MMA Shape (FP16)
  // ============================================================

  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  static constexpr int acc_per_warp_m = ACC_PER_WARP_M;
  static constexpr int acc_per_warp_n = ACC_PER_WARP_N;
  static constexpr int warp_k_iters   = WARP_K_ITERS;

  static constexpr int warps_per_block_m = WARPS_PER_BLOCK_M;
  static constexpr int warps_per_block_n = WARPS_PER_BLOCK_N;

  static_assert(is_pow2(acc_per_warp_m));
  static_assert(is_pow2(acc_per_warp_n));
  static_assert(is_pow2(warp_k_iters));
  static_assert(is_pow2(warps_per_block_m));
  static_assert(is_pow2(warps_per_block_n));

  // ============================================================
  // Tile Geometry
  // ============================================================

  static constexpr int WM = mma_m * acc_per_warp_m;
  static constexpr int WN = mma_n * acc_per_warp_n;
  static constexpr int BK = mma_k * warp_k_iters;
  static constexpr int block_k_iters = K/BK;

  static constexpr int BM = WM * warps_per_block_m;
  static constexpr int BN = WN * warps_per_block_n;

  static_assert(K % BK == 0);
  static_assert(M % BM == 0);
  static_assert(N % BN == 0);

  static constexpr int GM = M / BM;
  static constexpr int GN = N / BN;
  static_assert(GM == GN);

  // ============================================================
  // Warp Specialization
  // ============================================================

  static constexpr int num_compute_warps =
      warps_per_block_m * warps_per_block_n;

  static constexpr int producer_warp_id = num_compute_warps;

  static constexpr int num_warps =
      num_compute_warps + 1;

  static constexpr int block_size = num_warps * 32;
  static_assert(block_size <= 1024);

  // ============================================================
  // Persistent Scheduling
  // ============================================================

  static constexpr int num_tiles = GM * GN;
  static constexpr int num_sms   = NUM_SMS;

  static constexpr int persist_num_iters =
      (num_tiles + num_sms - 1) / num_sms;

  static constexpr int grid_size = num_sms;

  // ============================================================
  // Shared Memory (FP16 A/B/C)
  // ============================================================

  static constexpr int bk_stages = BK_STAGES;
  static constexpr int c_stages  = C_STAGES;

  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(__half);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(__half);

  static constexpr uint32_t Cs_bytes =
      BM * BN * sizeof(__half);

  static constexpr uint32_t smem_overhead =
      16 * (bk_stages + c_stages);

  static constexpr uint32_t shared_bytes =
      bk_stages * (As_bytes + Bs_bytes) +
      c_stages  * Cs_bytes +
      smem_overhead;

  // ============================================================
  // TMA Swizzle — A/B
  // ============================================================

  static constexpr uint32_t ld_bytes =
      BK * sizeof(__half);

  static_assert(ld_bytes <= 128);

  static constexpr CUtensorMapSwizzle ab_swizzle_mode =
      (ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                          CU_TENSOR_MAP_SWIZZLE_NONE;

  static_assert(ab_swizzle_mode != CU_TENSOR_MAP_SWIZZLE_NONE);

  static constexpr int ab_swizzle_num =
      (ab_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 128 :
      (ab_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 384 :
      (ab_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 896 :
                                                         0;

  // ============================================================
  // TMA Swizzle — C
  // ============================================================

  static constexpr uint32_t c_ld_bytes =
      BN * sizeof(__half);


  static constexpr CUtensorMapSwizzle c_swizzle_mode =
      (c_ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (c_ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (c_ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                            CU_TENSOR_MAP_SWIZZLE_NONE;

  static constexpr int c_swizzle_num =
      (c_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 128 :
      (c_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 384 :
      (c_swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 896 :
                                                        0;

};
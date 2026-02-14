#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

struct Sm120_BF16_Gemm_Config_swizzle_persist {

  static constexpr int M = 8192;
  static constexpr int N = 8192;
  static constexpr int K = 8192;

  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  static constexpr int apw_m = 4;
  static constexpr int apw_n = 4;
  static constexpr int wk_iters = 2;

  static constexpr int WM = mma_m * apw_m;
  static constexpr int WN = mma_n * apw_n;
  static constexpr int BK = mma_k * wk_iters;

  static constexpr int wpb_m = 2;
  static constexpr int wpb_n = 4;

  static constexpr int num_compute_warps = wpb_m * wpb_n;
  static constexpr int producer_warp_id  = num_compute_warps;
  static constexpr int num_warps_total   = num_compute_warps + 1;

  static constexpr int block_size = num_warps_total * 32;

  static constexpr int BM = WM * wpb_m;
  static constexpr int BN = WN * wpb_n;

  static constexpr int bk_iters  = K / BK;
  static constexpr int bk_stages = 2;

  static constexpr int GM = M / BM;
  static constexpr int GN = N / BN;

  // ============================================================
  // Persistent Scheduling Extension
  // ============================================================

  // Total number of C tiles in the problem
  static constexpr int num_tiles = GM * GN;

  // One resident block per SM (you will launch exactly this many)
  static constexpr int num_sms = 170;

  // Persistent iterations per SM:
  // ceil(num_tiles / num_sms)
  static constexpr int persist_num_iters =
      (num_tiles + num_sms - 1) / num_sms;

  // Grid launch size = num_sms (1 block per SM)
  static constexpr int grid_size = num_sms;

  // ============================================================

  static constexpr bool A_rowmajor = true;
  static constexpr bool B_colmajor = true;
  static constexpr bool C_rowmajor = true;

  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(nv_bfloat16);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(nv_bfloat16);

  static constexpr uint32_t shared_bytes =
      bk_stages * (As_bytes + Bs_bytes) + (8 * 1024);

  static_assert(shared_bytes <= 100 * 1024);

  static constexpr uint32_t ld_bytes =
      BK * sizeof(nv_bfloat16);

  static_assert(ld_bytes <= 128,
                "Leading dimension (bytes) must be <= 128B for swizzle.");

  // Pick swizzle mode directly by byte size
  static constexpr CUtensorMapSwizzle swizzle_mode =
      (ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                          CU_TENSOR_MAP_SWIZZLE_NONE;

  static_assert(swizzle_mode != CU_TENSOR_MAP_SWIZZLE_NONE,
                "Unsupported ld_bytes: must be exactly 32/64/128.");

  // Cute parameters (canonical)
  static constexpr int m_base  = 4;
  static constexpr int s_shift = 3;

  // b_bits matches swizzle size
  static constexpr int b_bits =
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 1 :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 2 :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 3 :
                                                     0;

  static_assert(b_bits != 0, "Invalid b_bits.");
};

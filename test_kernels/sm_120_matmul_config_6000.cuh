#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

struct Sm120_BF16_Gemm_Config {

  static constexpr int M = 8192;
  static constexpr int N = 8192;
  static constexpr int K = 8192;

  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  static constexpr int apw_m = 2;
  static constexpr int apw_n = 4;
  static constexpr int wk_iters = 4;

  static constexpr int WM = mma_m * apw_m;
  static constexpr int WN = mma_n * apw_n;
  static constexpr int BK = mma_k * wk_iters;

  static constexpr int wpb_m = 4;
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
  static constexpr int grid_size = GM * GN;

  static constexpr bool A_rowmajor = true;
  static constexpr bool B_colmajor = true;
  static constexpr bool C_rowmajor = true;

  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(nv_bfloat16);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(nv_bfloat16);

  static constexpr uint32_t shared_bytes =
      bk_stages * (As_bytes + Bs_bytes) + (2 * 1024);

  static_assert(shared_bytes <= 100 * 1024);
};

#pragma once

#include <cuda_bf16.h>

static constexpr bool is_pow2(int x)
{
  return x > 1 && ((x & (x - 1)) == 0);
}

static constexpr uint32_t cmax(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

struct AttnConfig
{
  static constexpr int mma_m = 16;
  static constexpr int mma_n = 16;
  static constexpr int mma_k = 16;

  // batch/head
  static constexpr int B = 4;
  static constexpr int H = 8;
  static constexpr int BH = B * H;

  static constexpr int D = 128;
  static constexpr int L_kv = 4096;
  static constexpr int L_q = 8192;

  static constexpr int block_L_q = 256;
  static constexpr int block_L_kv = 32;

  static constexpr int warp_L_q =  32;
  static constexpr int warp_L_kv = 32;

  static constexpr int num_warps = block_L_q / warp_L_q;
  static constexpr int block_size = num_warps * 32;

  static constexpr uint32_t Ks_bytes =
      block_L_kv * D * sizeof(nv_bfloat16);

  static constexpr uint32_t Qs_bytes =
      block_L_q * D * sizeof(nv_bfloat16);

  static constexpr uint32_t Vs_bytes =
      block_L_kv * D * sizeof(nv_bfloat16);

  static constexpr uint32_t shared_bytes =
      cmax(Qs_bytes, Ks_bytes + Vs_bytes) + 16;

  // 1D grid

  static constexpr int GL_q = L_q/block_L_q;

  static constexpr int grid_size = BH*GL_q;

  static_assert(D % mma_k == 0);
  static_assert(warp_L_q % mma_m == 0);
  static_assert(warp_L_kv % mma_n == 0);
  static_assert(block_L_q % warp_L_q == 0);
  static_assert(block_L_kv % warp_L_kv == 0);
};
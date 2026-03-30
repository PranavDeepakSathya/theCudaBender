#pragma once

#include <cuda_bf16.h>

#ifndef BLOCK_L_Q
#define BLOCK_L_Q 128
#endif
#ifndef BLOCK_L_KV
#define BLOCK_L_KV 64
#endif

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

  static constexpr int B  = 4;
  static constexpr int H  = 8;
  static constexpr int BH = B * H;

  static constexpr int D    = 128;
  static constexpr int L_kv = 4096;
  static constexpr int L_q  = 8192;

  static constexpr int block_L_q  = BLOCK_L_Q;
  static constexpr int block_L_kv = BLOCK_L_KV;

  static constexpr int warp_L_q = 32;

  static_assert(block_L_q  % warp_L_q == 0);
  static_assert(block_L_kv % mma_n    == 0);
  static_assert(L_q  % block_L_q  == 0);
  static_assert(L_kv % block_L_kv == 0);

  static constexpr int num_warps  = block_L_q / warp_L_q;
  static constexpr int block_size = num_warps * 32;

  static_assert(block_size < 1024);

  static constexpr uint32_t Ks_bytes = block_L_kv * D * sizeof(nv_bfloat16);
  static constexpr uint32_t Qs_bytes = block_L_q  * D * sizeof(nv_bfloat16);
  static constexpr uint32_t Vs_bytes = block_L_kv * D * sizeof(nv_bfloat16);

  static constexpr uint32_t Ks_total = Ks_bytes * 2;

  static constexpr uint32_t shared_bytes =
      cmax(Qs_bytes, Ks_total) + 8 + (2*16); // q_bar(8) + 2*k_bar(16) + 2*v_bar(16)

  static constexpr uint32_t bar_start = cmax(Qs_bytes, Ks_total);

  static constexpr int GL_q      = L_q / block_L_q;
  static constexpr int grid_size = BH * GL_q;

  static constexpr uint32_t ld_bytes = block_L_kv * sizeof(nv_bfloat16);

  static constexpr CUtensorMapSwizzle swizzle_mode =
      (ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                          CU_TENSOR_MAP_SWIZZLE_NONE;

  static constexpr int swizzle_num =
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 128 :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 384 :
      (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 896 : 0;

  static constexpr CUtensorMapSwizzle D_swizzle_mode = CU_TENSOR_MAP_SWIZZLE_128B;
  static constexpr int D_swizzle_num = 896;
};

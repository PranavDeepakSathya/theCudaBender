#pragma once

#include <cuda_bf16.h>

static constexpr bool is_pow2(int x)
{
  return x > 1 && ((x & (x - 1)) == 0);
}

struct AttnConfig
{
  static constexpr int mma_m = 16; 
  static constexpr int mma_n = 16; 
  static constexpr int mma_k = 16; 
  
  static constexpr int D = 128; 
  static constexpr int L_kv = 4096;
  static constexpr int L_q = 4096;
  static constexpr int block_L_q = 128; 
  static constexpr int block_L_kv = 128; 
  static constexpr int warp_L_q = 32; 
  static constexpr int warp_L_kv = 32; 

  static constexpr int num_warps = block_L_q/warp_L_q; 
  static constexpr int block_size = num_warps*32; 
  static constexpr uint32_t Ks_bytes = block_L_kv*D*sizeof(nv_bfloat16); 
  static constexpr uint32_t Qs_bytes = block_L_q*D*sizeof(nv_bfloat16); 
  static constexpr uint32_t Vs_bytes = block_L_kv*D*sizeof(nv_bfloat16); 

  static constexpr uint32_t shared_bytes = Qs_bytes + Ks_bytes + Vs_bytes + 1024;
  static constexpr int grid_size = L_q/block_L_q;
  static_assert(D % mma_k == 0);
  static_assert(warp_L_q % mma_m == 0);
  static_assert(warp_L_kv % mma_n == 0);
  static_assert(block_L_q % warp_L_q == 0);
  static_assert(block_L_kv % warp_L_kv == 0);

};
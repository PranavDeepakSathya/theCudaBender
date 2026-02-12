#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

struct sm120_single_buffer_gemm_config
{
  static constexpr int M = 4096; 
  static constexpr int N = 4096; 
  static constexpr int K = 4096; 

  static constexpr int MMA_M = 16;
  static constexpr int MMA_N = 8;
  static constexpr int MMA_K = 16;

  static constexpr int ACC_PER_WARP_M = 2; 
  static constexpr int ACC_PER_WARP_N = 2; 
  static constexpr int MMA_K_ITERS_PER_WARP = 2; 

  static constexpr int WARP_M = MMA_M*ACC_PER_WARP_M; 
  static constexpr int WARP_N = MMA_N*ACC_PER_WARP_N; 
  static constexpr int BLOCK_K = MMA_K_ITERS_PER_WARP*MMA_K; 

  static constexpr int WARPS_PER_BLOCK_M = 2;
  static constexpr int WARPS_PER_BLOCK_N = 2; 

  static constexpr int BLOCK_M = WARP_M*WARPS_PER_BLOCK_M; 
  static constexpr int BLOCK_N = WARP_N*WARPS_PER_BLOCK_N;

  static constexpr int GRID_M = M/BLOCK_M; 
  static constexpr int GRID_N = N/BLOCK_N; 
  static constexpr int BLOCK_K_ITERS = K/BLOCK_K; 

  static constexpr int NUM_WARPS =
    WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;

  static constexpr int BLOCK_THREADS = NUM_WARPS * 32;
  static constexpr int SMEM_A_ELEMS = BLOCK_M * BLOCK_K;
  static constexpr int SMEM_B_ELEMS = BLOCK_K * BLOCK_N;

  static constexpr int GRID_BLOCKS = GRID_M*GRID_N;

  static constexpr uint32_t SMEM_A_BYTES = SMEM_A_ELEMS * sizeof(nv_bfloat16);
  static constexpr uint32_t SMEM_B_BYTES = SMEM_B_ELEMS * sizeof(nv_bfloat16);
  static constexpr uint32_t SMEM_PAD_BYTES = 4 * 1024;

  static constexpr uint32_t SMEM_BYTES_TOTAL =
  SMEM_A_BYTES + SMEM_B_BYTES + SMEM_PAD_BYTES;

  static_assert(M % BLOCK_M == 0);
  static_assert(N % BLOCK_N == 0);
  static_assert(K % BLOCK_K == 0);

  static_assert(BLOCK_K % MMA_K == 0);
  static_assert(BLOCK_M % MMA_M == 0);
  static_assert(BLOCK_N % MMA_N == 0);
};
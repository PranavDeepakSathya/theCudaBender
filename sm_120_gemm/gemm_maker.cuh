#pragma once 
#include "../clean_abstractions/all.cuh"

template <typename Cfg>
struct GemmMaker
{
  const int lane_id;
  const int warp_id;
  const int thread_id;
  const CUtensorMap* tma_a;
  const CUtensorMap* tma_b;
  int c2_row_start;
  int c2_col_start;
  uint32_t a_s2r_off;
  uint32_t b_s2r_off;

  __device__ __forceinline__
  GemmMaker(const CUtensorMap& tma_a_, const CUtensorMap& tma_b_, int tid)
    : tma_a(&tma_a_), tma_b(&tma_b_),
      thread_id(tid),
      lane_id(tid % 32),
      warp_id(tid / 32)
  {
    int warp_start_m = (warp_id / Cfg::warps_per_block_n) * Cfg::WM;
    int warp_start_n = (warp_id % Cfg::warps_per_block_n) * Cfg::WN;

    a_s2r_off = ((warp_start_m + (lane_id % 16)) * Cfg::BK + (8 * (lane_id / 16))) * sizeof(nv_bfloat16);
    b_s2r_off = ((warp_start_n + ((lane_id % 8) + (8 * (lane_id / 16)))) * Cfg::BK + (8 * ((lane_id / 8) % 2))) * sizeof(nv_bfloat16);
    c2_row_start = warp_start_m + (lane_id/4);
    c2_col_start = warp_start_n + 2*(lane_id%4);
  }

    __device__ __forceinline__
  void load_A_g2s(int bk_idx, int block_start_m, uint32_t As_addr, uint32_t mbar_addr)
  {

    cp_async_bulk_tensor_2d(As_addr, tma_a, bk_idx * Cfg::BK, block_start_m, mbar_addr);

  }

  __device__ __forceinline__
  void load_B_g2s(int bk_idx, int block_start_n, uint32_t Bs_addr, uint32_t mbar_addr)
  {
    cp_async_bulk_tensor_2d(Bs_addr, tma_b, bk_idx * Cfg::BK, block_start_n, mbar_addr);
  }

  __device__ __forceinline__
  void store_C(float2* C2, float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4], int block_start_m, int block_start_n)
  {
    static constexpr int ldc2 = Cfg::N / 2;
    int C_row_start = block_start_m + c2_row_start;
    int C_col_start = block_start_n + c2_col_start;

    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      #pragma unroll
      for (int n = 0; n < Cfg::acc_per_warp_n; n++)
      {
        int C_row = C_row_start + (m * Cfg::mma_m);
        int C_col = (C_col_start + (n * Cfg::mma_n)) / 2;
        C2[C_row       * ldc2 + C_col] = {rc[m][n][0], rc[m][n][1]};
        C2[(C_row + 8) * ldc2 + C_col] = {rc[m][n][2], rc[m][n][3]};
      }
    }
  }

    __device__ __forceinline__
  void load_A_s2r(uint32_t ra[Cfg::acc_per_warp_m][4], uint32_t As_addr, int k)
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      uint32_t addr = As_addr + compact_swizzle<Cfg::swizzle_num>(
        a_s2r_off + ((m * Cfg::mma_m * Cfg::BK) + (k * Cfg::mma_k)) * sizeof(nv_bfloat16));
      ldmatrix_m8n8_x4_b16(ra[m], addr);
    }
  }

  __device__ __forceinline__
  void load_B_s2r(uint32_t rb[Cfg::acc_per_warp_n][2], uint32_t Bs_addr, int k)
  {
    #pragma unroll
    for (int n = 0; n < Cfg::acc_per_warp_n / 2; n++)
    {
      uint32_t addr = Bs_addr + compact_swizzle<Cfg::swizzle_num>(
        b_s2r_off + ((2 * n * Cfg::mma_n * Cfg::BK) + (k * Cfg::mma_k)) * sizeof(nv_bfloat16));
      ldmatrix_m8n8_x4_b16(rb[2 * n], addr);
    }
  }

    __device__ __forceinline__
  void mma(float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4],
          uint32_t ra[Cfg::acc_per_warp_m][4],
          uint32_t rb[Cfg::acc_per_warp_n][2])
  {
    #pragma unroll
    for (int m = 0; m < Cfg::acc_per_warp_m; m++)
    {
      #pragma unroll
      for (int n = 0; n < Cfg::acc_per_warp_n; n++)
      {
        mma_m16n8k16_row_col_f32_bf16(rc[m][n], ra[m], rb[n]);
      }
    }
  }

};
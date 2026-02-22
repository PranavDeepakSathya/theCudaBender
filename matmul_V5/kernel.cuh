#pragma once

#include "../atoms/all.cuh"
#include "config.cuh"
#include "smem_allocator.cuh"

namespace ptx = cuda::ptx;
namespace wa = warp_function;

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C
)

{


  extern __shared__ __align__(1024) uint8_t smem_raw[];
  SmemAllocator<Cfg> smem(smem_raw);
  int t = threadIdx.x; 
  int b = blockIdx.x; 
  int w = t / 32;
  int l = t % 32;
  int warp_start_m = (w/Cfg::warps_per_block_n)*Cfg::WM;
  int warp_start_n = (w%Cfg::warps_per_block_n)*Cfg::WN; 
  int C_row_start = warp_start_m + (l/4);
  int C_col_start = warp_start_n + (2*(l%4));
  float2* C2 = reinterpret_cast<float2*>(C); 
  int ldc2 = Cfg::N/2;

  uint32_t as_ld_base = ((warp_start_m + (l%16))*Cfg::BK + (8*(l/16)))*sizeof(nv_bfloat16);
  uint32_t bs_ld_base = ((warp_start_n + (l%8))*Cfg::BK + (8*(l/8)))*sizeof(nv_bfloat16);



  if (t == 0)
  { 
    #pragma unroll
    for (int stage = 0; stage < Cfg::bk_stages; stage++)
    {
      mbarrier_init(smem.full(stage), 1);
      mbarrier_init(
          smem.empty(stage),
          Cfg::warps_per_block_m *
          Cfg::warps_per_block_n * 32
      );
    }

    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (w == Cfg::producer_warp_id)
  {
    int stage = 0;
    int phase = 1;

    for (int iter = 0; iter < Cfg::persist_num_iters; iter++)
    {
      int tile_id = tile_sched::get_linear_tile_id<Cfg::num_tiles,Cfg::num_sms>(iter,b);

      if(tile_id == -1) break;
      uint32_t m, n;
      tile_sched::morton_decode_2d((uint32_t)tile_id, m, n);
      if (m >= Cfg::GM || n >= Cfg::GN) {
        continue;   // skip invalid morton tile
      }
      int block_start_m = m*Cfg::BM;
      int block_start_n = n*Cfg::BN;
      #pragma unroll
      for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; ++bk_idx, ++stage)
      {
        if (stage == Cfg::bk_stages)
        {
          stage = 0;
          phase ^= 1;
        }

        mbarrier_wait_parity(smem.empty(stage), phase);
        tma_fence();
        if (l == 0)
        {
          mbarrier_arrive_expect_tx(smem.full(stage), Cfg::As_bytes + Cfg::Bs_bytes);
          cp_async_bulk_tensor_2d(smem.A(stage),&a_map,bk_idx*Cfg::BK, block_start_m,smem.full(stage));
          cp_async_bulk_tensor_2d(smem.B(stage),&b_map,bk_idx*Cfg::BK, block_start_n,smem.full(stage));
        }
      }
    }
  }

  else
  {
    int stage = 0;
    int phase = 0;

    #pragma unroll
    for (int iter = 0; iter < Cfg::persist_num_iters; iter++)
    {
      int tile_id = tile_sched::get_linear_tile_id<Cfg::num_tiles,Cfg::num_sms>(iter,b);

      if(tile_id == -1) break;
      uint32_t m, n;
      tile_sched::morton_decode_2d((uint32_t)tile_id, m, n);
      if (m >= Cfg::GM || n >= Cfg::GN) {
        continue;   // skip invalid morton tile
      }
      
      int block_start_m = m*Cfg::BM;
      int block_start_n = n*Cfg::BN;
      
      int C_row_block = C_row_start + block_start_m;
      int C_col_block = C_col_start + block_start_n; 
      uint32_t ra[Cfg::acc_per_warp_m][Cfg::warp_k_iters][4];
      uint32_t rb[Cfg::acc_per_warp_n][Cfg::warp_k_iters][2];
      float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0}; 

      #pragma unroll
      for (int bk_idx = 0; bk_idx < Cfg::block_k_iters; ++bk_idx, ++stage)
      {
        if (stage == Cfg::bk_stages)
        {
          stage = 0;
          phase ^= 1;
        }

        mbarrier_wait_parity(smem.full(stage), phase);
        #pragma unroll
        for (int m = 0; m < Cfg::acc_per_warp_m; m++)
        {
          #pragma unroll
          for (int k = 0; k < Cfg::warp_k_iters; k++)
          {
            uint32_t a_shared_offset = compact_swizzle<Cfg::swizzle_num>(as_ld_base + ((m*Cfg::mma_m*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
            wa::ldmatrix_m8n8_x4_b16(ra[m][k], smem.A(stage) + a_shared_offset);
          }
        }
        #pragma unroll
        for (int n = 0; n < Cfg::acc_per_warp_n; n++)
        {
          #pragma unroll
          for (int k = 0; k < Cfg::warp_k_iters; k+=2)
          {
            uint32_t b_shared_offset = compact_swizzle<Cfg::swizzle_num>(bs_ld_base + ((n*Cfg::mma_n*Cfg::BK) + (k*Cfg::mma_k))*sizeof(nv_bfloat16));
            wa::ldmatrix_m8n8_x4_b16(rb[n][k], smem.B(stage) + b_shared_offset);
          }
        }

        
        #pragma unroll
        for (int k = 0; k < Cfg::warp_k_iters; k++)
        {
          #pragma unroll
          for (int m = 0; m < Cfg::acc_per_warp_m; m++)
          {
            #pragma unroll
            for (int n = 0; n < Cfg::acc_per_warp_n; n++)
            {
              wa::mma_m16n8k16_row_col_f32_bf16(rc[m][n], ra[m][k], rb[n][k]);
            }
          }
        }

        mbarrier_arrive(smem.empty(stage));

      }
      #pragma unroll
      for (int m = 0; m < Cfg::acc_per_warp_m; m++)
      {
        #pragma unroll
        for (int n = 0; n < Cfg::acc_per_warp_n; n++)
        {
          int C_row = C_row_block + (m*Cfg::mma_m);
          int C_col = (C_col_block + (n*Cfg::mma_n))/2;
          float2 v0 = {rc[m][n][0], rc[m][n][1]}; 
          float2 v1 = {rc[m][n][2], rc[m][n][3]}; 
          C2[(C_row)*ldc2 + (C_col)] = v0; 
          C2[(C_row+8)*ldc2 + (C_col)] = v1;
        }
      }
    }
  }
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev
)

{
  launcher.launch(matmul_kernel<Cfg>, a_map,b_map,C_dev);
}
#pragma once
#include "../atoms/all.cuh"
#include <math.h>
namespace wa = warp_function;



template <class Cfg>
__global__ void attention_kernel(__grid_constant__ const CUtensorMap q_map,
                                 __grid_constant__ const CUtensorMap k_map,
                                __grid_constant__ const CUtensorMap v_map,
                                float*O)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); 
  uint32_t Qs = smem; 
  uint32_t Ks = smem; //same smem for both, as we will hold Qs in regs throughout the whole thing
  uint32_t Vs = Ks + Cfg::w_K_bytes; 
  uint32_t bar = Vs + Cfg::w_V_bytes; 
  uint32_t rq[4]; 
  uint32_t rk[2];
  float rs[4] = {0.0}; 
  float maxim[2] = {-INFINITY}; //each lane holds the max and rows sum of row id l/4, l/4 + 8
  float row_sum[2] = {0.0};
  uint32_t rv[2];
  float ro[4] = {0.0};

  int l = threadIdx.x; 
  if (l == 0) mbarrier_init(bar, 32); 
  __syncthreads(); 
  if (l == 0)
  {
    //copy Qs 
    cp_async_bulk_tensor_2d(Qs, &q_map, 0,0,bar);
    mbarrier_arrive_expect_tx(bar, Cfg::w_Q_bytes); 

  }
  else mbarrier_arrive(bar); 
  mbarrier_wait_parity(bar,0); 
  uint32_t Qs_addr =  Qs + (((l%16)*Cfg::w_d + (8*(l/16)))*sizeof(nv_bfloat16)); 
  wa::ldmatrix_m8n8_x4_b16(rq,Qs_addr); 
  __syncthreads(); 
  tma_fence(); 

  if (l == 0)
  {
    cp_async_bulk_tensor_2d(Ks, &k_map, 0,0, bar);
    cp_async_bulk_tensor_2d(Vs, &v_map, 0,0, bar);
    mbarrier_arrive_expect_tx(bar, Cfg::w_K_bytes + Cfg::w_V_bytes);
  }
  else mbarrier_arrive(bar); 
  mbarrier_wait_parity(bar,1); 
  uint32_t Ks_addr = Ks + (((l%8)*Cfg::w_d +(8*(l/8)))*sizeof(nv_bfloat16));
  wa::ldmatrix_m8n8_x2_b16(rk, Ks_addr); 

  wa::mma_m16n8k16_row_col_f32_bf16(rs,rq,rk); 

  maxim[0] = fmaxf(rs[0], rs[1]);
  maxim[1] = fmaxf(rs[2], rs[3]);

  // 4-lane butterfly for each row group
  maxim[0] = fmaxf(maxim[0], __shfl_xor_sync(0xffffffff, maxim[0], 2));
  maxim[0] = fmaxf(maxim[0], __shfl_xor_sync(0xffffffff, maxim[0], 1));

  maxim[1] = fmaxf(maxim[1], __shfl_xor_sync(0xffffffff, maxim[1], 2));
  maxim[1] = fmaxf(maxim[1], __shfl_xor_sync(0xffffffff, maxim[1], 1));

  row_sum[0] =
      __expf(rs[0] - maxim[0]) +
      __expf(rs[1] - maxim[0]);

  row_sum[1] =
      __expf(rs[2] - maxim[1]) +
      __expf(rs[3] - maxim[1]);

  row_sum[0] += __shfl_xor_sync(0xffffffff, row_sum[0], 2);
  row_sum[0] += __shfl_xor_sync(0xffffffff, row_sum[0], 1);

  row_sum[1] += __shfl_xor_sync(0xffffffff, row_sum[1], 2);
  row_sum[1] += __shfl_xor_sync(0xffffffff, row_sum[1], 1);

  float rp[4];

  rp[0] = __expf(rs[0] - maxim[0]) / row_sum[0];
  rp[1] = __expf(rs[1] - maxim[0]) / row_sum[0];

  rp[2] = __expf(rs[2] - maxim[1]) / row_sum[1];
  rp[3] = __expf(rs[3] - maxim[1]) / row_sum[1];

  __nv_bfloat162 pair0 = __float22bfloat162_rn({rp[0], rp[1]});
  __nv_bfloat162 pair1 = __float22bfloat162_rn({rp[2], rp[3]});
  uint32_t rp_16[4];
  rp_16[0] = reinterpret_cast<uint32_t&>(pair0);
  rp_16[1] = 0u;
  rp_16[2] = reinterpret_cast<uint32_t&>(pair1);
  rp_16[3] = 0u;

  uint32_t Vs_addr = Vs + (((l%8)*Cfg::w_lk +(8*(l/8)))*sizeof(nv_bfloat16));
  wa::ldmatrix_m8n8_x2_b16(rv, Vs_addr); 
  wa::mma_m16n8k16_row_col_f32_bf16(ro,rp_16,rv); 
  float2* O2 = reinterpret_cast<float2*>(O); 
  int ldO2 = Cfg::D/2; 
  float2 v0 = {ro[0],ro[1]}; 
  float2 v1 = {ro[2],ro[3]}; 
  int lane_row = l/4; 
  int lane_col = (l%4);
  O2[lane_row*ldO2 + lane_col] = v0; 
  O2[(lane_row+8)*ldO2 + lane_col] = v1;
}


template <class Cfg>
inline void launch_attention(
    NaiveLauncher& launcher,
    CUtensorMap q_map,
    CUtensorMap k_map,
    CUtensorMap v_map,
    float* O)
{
    launcher.launch(attention_kernel<Cfg>, q_map, k_map, v_map, O);
}
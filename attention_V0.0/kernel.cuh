#pragma once
#include "../atoms/all.cuh"
#include <math.h>
namespace wa = warp_function;

__device__ __forceinline__
void flash_softmax_update(
    float s_frag[8],
    float o_frag[8],
    float m_i[2],
    float l_i[2])
{
    const unsigned mask = 0xffffffff;

    // lane-local maxima
    float row0_local =
        fmaxf(fmaxf(s_frag[0], s_frag[1]),
              fmaxf(s_frag[4], s_frag[5]));

    float row1_local =
        fmaxf(fmaxf(s_frag[2], s_frag[3]),
              fmaxf(s_frag[6], s_frag[7]));

    // 4-lane reductions
    row0_local = fmaxf(row0_local, __shfl_xor_sync(mask,row0_local,1));
    row0_local = fmaxf(row0_local, __shfl_xor_sync(mask,row0_local,2));

    row1_local = fmaxf(row1_local, __shfl_xor_sync(mask,row1_local,1));
    row1_local = fmaxf(row1_local, __shfl_xor_sync(mask,row1_local,2));

    float m0_new = fmaxf(m_i[0], row0_local);
    float m1_new = fmaxf(m_i[1], row1_local);

    float alpha0 = __expf(m_i[0] - m0_new);
    float alpha1 = __expf(m_i[1] - m1_new);

    // scale previous output
    o_frag[0] *= alpha0;
    o_frag[1] *= alpha0;
    o_frag[4] *= alpha0;
    o_frag[5] *= alpha0;

    o_frag[2] *= alpha1;
    o_frag[3] *= alpha1;
    o_frag[6] *= alpha1;
    o_frag[7] *= alpha1;

    // exponentiate scores
    float row0_sum = 0.f;
    float row1_sum = 0.f;

    s_frag[0] = __expf(s_frag[0] - m0_new);
    s_frag[1] = __expf(s_frag[1] - m0_new);
    s_frag[4] = __expf(s_frag[4] - m0_new);
    s_frag[5] = __expf(s_frag[5] - m0_new);

    row0_sum = s_frag[0] + s_frag[1] + s_frag[4] + s_frag[5];

    s_frag[2] = __expf(s_frag[2] - m1_new);
    s_frag[3] = __expf(s_frag[3] - m1_new);
    s_frag[6] = __expf(s_frag[6] - m1_new);
    s_frag[7] = __expf(s_frag[7] - m1_new);

    row1_sum = s_frag[2] + s_frag[3] + s_frag[6] + s_frag[7];

    // row sum reductions
    row0_sum += __shfl_xor_sync(mask,row0_sum,1);
    row0_sum += __shfl_xor_sync(mask,row0_sum,2);

    row1_sum += __shfl_xor_sync(mask,row1_sum,1);
    row1_sum += __shfl_xor_sync(mask,row1_sum,2);

    l_i[0] = l_i[0]*alpha0 + row0_sum;
    l_i[1] = l_i[1]*alpha1 + row1_sum;

    m_i[0] = m0_new;
    m_i[1] = m1_new;
}

__device__ __forceinline__
void pack_p_frag(
    const float s_frag[8],
    uint32_t p_frag_packed[4])
{
    nv_bfloat162 p0 = __float22bfloat162_rn(s_frag[0], s_frag[1]);
    nv_bfloat162 p1 = __float22bfloat162_rn(s_frag[2], s_frag[3]);
    nv_bfloat162 p2 = __float22bfloat162_rn(s_frag[4], s_frag[5]);
    nv_bfloat162 p3 = __float22bfloat162_rn(s_frag[6], s_frag[7]);

    p_frag_packed[0] = reinterpret_cast<uint32_t&>(p0);
    p_frag_packed[1] = reinterpret_cast<uint32_t&>(p1);
    p_frag_packed[2] = reinterpret_cast<uint32_t&>(p2);
    p_frag_packed[3] = reinterpret_cast<uint32_t&>(p3);
}


template <class Cfg>
__global__ void attention_kernel(__grid_constant__ const CUtensorMap q_map,
                                 __grid_constant__ const CUtensorMap k_map,
                                __grid_constant__ const CUtensorMap v_map,
                                float*O)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); 
  uint32_t Qs = smem; 
  uint32_t Ks = smem;
  uint32_t Vs = smem + (Cfg::Ks_bytes); 
  uint32_t Qs_bar = Vs + (Cfg::Vs_bytes); 
  uint32_t KVs_bar = Qs_bar + 16; 



  int b = blockIdx.x; 
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 

  if (t == 0)
  { 
    mbarrier_init(Qs_bar,Cfg::block_size); 
    mbarrier_init(KVs_bar,Cfg::block_size); 
  }
  __syncthreads();


  int block_start_lq = b*block_L_q; 
  int warp_start_lq = w*warp_L_q; 

  uint32_t q_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::D/Cfg::mma_k][4]; 
  uint32_t kt_frag[Cfg::warp_L_kv/Cfg::mma_n][Cfg::D/Cfg::mma_k][4]; 
  float s_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::warp_L_kv/Cfg::mma_n][8];
  uint32_t p_frag_packed[Cfg::warp_L_q/Cfg::mma_m][Cfg::warp_L_kv/Cfg::mma_k][4];
  uint32_t v_frag[Cfg::D/Cfg::mma_n][Cfg::warp_L_kv/Cfg::mma_k][4]; 
  float o_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::D/Cfg::mma_n][8]; 
  float m_i[Cfg::warp_L_q/Cfg::mma_m][2]; 
  float l_i[Cfg::warp_L_q/Cfg::mma_m[2]; 


  if (t == 0)
  {
    cp_async_bulk_tensor_2d(Qs,&q_map,0,block_start_lq,Qs_bar)
    mbarrier_arrive_expect_tx(Qs_bar, Cfg::Qs_bytes);
  }
  else mbarrier_arrive(Qs_bar); 

  mbarrier_wait_parity(Qs_bar,0); 

  for (int lq_idx = 0; lq_idx < Cfg::warp_L_q/Cfg::mma_m; lq_idx++)
  {
    for (int d_idx = 0; d_idx < Cfg::D/Cfg::mma_k; d_idx++)
    {
      uint32_t q_frag_ld_addr = Qs + ((warp_start_lq + (lq_idx*Cfg::mma_m) + (l%16))*Cfg::D + ((d_idx*Cfg::mma_k + (8*(l/16)))))*sizeof(nv_bfloat16);
      wa::ldmatrix_m8n8_x4_b16(q_frag[lq_idx][d_idx],q_frag_ld_addr);
    }
  }
  __syncthreads(); 

  int parity = 0; 

  for (int blkv = 0; blkv < Cfg::L_kv/Cfg::block_L_kv; blkv++)
  {
    if (t == 0)
    {
      cp_async_bulk_tensor_2d(Ks,&k_map,0,blkv*Cfg::block_L_kv,KVs_bar);
      cp_async_bulk_tensor_2d(Vs,&v_map,blkv*Cfg::block_L_kv,0,KVs_bar);
      mbarrier_arrive_expect_tx(KVs_bar, Cfg::Ks_bytes + Cfg::Vs_bytes)
    }
    else mbarrier_arrive(KVs_bar); 

    mbarrier_wait_parity(KVs_bar,parity);
    parity ^=1; 


    for (int wlkv = 0; wlkv < Cfg::block_L_kv/Cfg::warp_L_kv; wlkv++)
    {

    }


  }


 
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
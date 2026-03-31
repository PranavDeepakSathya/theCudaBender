#pragma once
#include "../atoms/all.cuh"
#include <math.h>
namespace wa = warp_function;

template <class Cfg>
__device__ __forceinline__
void flash_softmax_update(
    float s_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::block_L_kv/Cfg::mma_n][8],
    float o_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::D/Cfg::mma_n][8],
    float m_i[Cfg::warp_L_q/Cfg::mma_m][2],
    float l_i[Cfg::warp_L_q/Cfg::mma_m][2])
{
    const unsigned mask = 0xffffffff;

    #pragma unroll
    for (int lq = 0; lq < Cfg::warp_L_q/Cfg::mma_m; lq++)
    {
        float row0_max = -INFINITY;
        float row1_max = -INFINITY;

        //--------------------------------
        // first pass: max over lkv
        //--------------------------------
        #pragma unroll
        for (int lkv = 0; lkv < Cfg::block_L_kv/Cfg::mma_n; lkv++)
        {
            float r0 =
                fmaxf(fmaxf(s_frag[lq][lkv][0], s_frag[lq][lkv][1]),
                      fmaxf(s_frag[lq][lkv][4], s_frag[lq][lkv][5]));

            float r1 =
                fmaxf(fmaxf(s_frag[lq][lkv][2], s_frag[lq][lkv][3]),
                      fmaxf(s_frag[lq][lkv][6], s_frag[lq][lkv][7]));

            row0_max = fmaxf(row0_max, r0);
            row1_max = fmaxf(row1_max, r1);
        }

        //--------------------------------
        // warp reduction (row group)
        //--------------------------------
        row0_max = fmaxf(row0_max, __shfl_xor_sync(mask,row0_max,1));
        row0_max = fmaxf(row0_max, __shfl_xor_sync(mask,row0_max,2));

        row1_max = fmaxf(row1_max, __shfl_xor_sync(mask,row1_max,1));
        row1_max = fmaxf(row1_max, __shfl_xor_sync(mask,row1_max,2));

        float m0_new = fmaxf(m_i[lq][0], row0_max);
        float m1_new = fmaxf(m_i[lq][1], row1_max);

        float alpha0 = __expf(m_i[lq][0] - m0_new);
        float alpha1 = __expf(m_i[lq][1] - m1_new);

        //--------------------------------
        // scale output fragments
        //--------------------------------
        #pragma unroll
        for (int d = 0; d < Cfg::D/Cfg::mma_n; d++)
        {
            float *o = o_frag[lq][d];

            o[0] *= alpha0; o[1] *= alpha0;
            o[4] *= alpha0; o[5] *= alpha0;

            o[2] *= alpha1; o[3] *= alpha1;
            o[6] *= alpha1; o[7] *= alpha1;
        }

        //--------------------------------
        // second pass: exp + sum
        //--------------------------------
        float row0_sum = 0.f;
        float row1_sum = 0.f;

        #pragma unroll
        for (int lkv = 0; lkv < Cfg::block_L_kv/Cfg::mma_n; lkv++)
        {
            float *s = s_frag[lq][lkv];

            s[0] = __expf(s[0] - m0_new);
            s[1] = __expf(s[1] - m0_new);
            s[4] = __expf(s[4] - m0_new);
            s[5] = __expf(s[5] - m0_new);

            row0_sum += s[0] + s[1] + s[4] + s[5];

            s[2] = __expf(s[2] - m1_new);
            s[3] = __expf(s[3] - m1_new);
            s[6] = __expf(s[6] - m1_new);
            s[7] = __expf(s[7] - m1_new);

            row1_sum += s[2] + s[3] + s[6] + s[7];
        }

        //--------------------------------
        // warp reduction for sums
        //--------------------------------
        row0_sum += __shfl_xor_sync(mask,row0_sum,1);
        row0_sum += __shfl_xor_sync(mask,row0_sum,2);

        row1_sum += __shfl_xor_sync(mask,row1_sum,1);
        row1_sum += __shfl_xor_sync(mask,row1_sum,2);

        //--------------------------------
        // update state
        //--------------------------------
        l_i[lq][0] = l_i[lq][0]*alpha0 + row0_sum;
        l_i[lq][1] = l_i[lq][1]*alpha1 + row1_sum;

        m_i[lq][0] = m0_new;
        m_i[lq][1] = m1_new;
    }
}

template<class Cfg>
__device__ __forceinline__
void pack_p_frag(
    const float s_frag[Cfg::warp_L_q/Cfg::mma_m]
                     [Cfg::block_L_kv/Cfg::mma_n][8],
    uint32_t p_frag_packed[Cfg::block_L_kv/Cfg::mma_k][Cfg::warp_L_q/Cfg::mma_m][4])
{
  #pragma unroll
  for (int lq = 0; lq < Cfg::warp_L_q / Cfg::mma_m; lq++)
  {
  #pragma unroll
    for (int lkv = 0; lkv < Cfg::block_L_kv / Cfg::mma_n; lkv++)
    {
      const float* s = s_frag[lq][lkv];
      uint32_t* p = p_frag_packed[lkv][lq];

      nv_bfloat162 p0 = __float22bfloat162_rn({s[0], s[1]});
      nv_bfloat162 p1 = __float22bfloat162_rn({s[2], s[3]});
      nv_bfloat162 p2 = __float22bfloat162_rn({s[4], s[5]});
      nv_bfloat162 p3 = __float22bfloat162_rn({s[6], s[7]});

      p[0] = reinterpret_cast<uint32_t&>(p0);
      p[1] = reinterpret_cast<uint32_t&>(p1);
      p[2] = reinterpret_cast<uint32_t&>(p2);
      p[3] = reinterpret_cast<uint32_t&>(p3);
    }
  }
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
  uint32_t Vs = smem + Cfg::Ks_bytes; 
  uint32_t q_bar = smem + Cfg::bar_start; 
  uint32_t k_bar = q_bar + 8; 
  uint32_t v_bar = k_bar + 8; 

  int b = blockIdx.x; 
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 

  if (t == 0)
  {
    mbarrier_init(q_bar, 32); 
    mbarrier_init(k_bar, 32);
    mbarrier_init(v_bar, 32);
    asm volatile("fence.mbarrier_init.release.cluster;"); 
  }
  __syncthreads(); 

  int block_start_lq = (b%Cfg::GL_q)*Cfg::block_L_q; 
  int block_start_BH = (b/Cfg::GL_q);
  int warp_start_lq = w*Cfg::warp_L_q; 

  uint32_t q_frag[Cfg::D/Cfg::mma_k][Cfg::warp_L_q/Cfg::mma_m][4]; 
  uint32_t kt_frag[Cfg::D/Cfg::mma_k][Cfg::block_L_kv/Cfg::mma_n][4]; 
  
  uint32_t p_frag_packed[Cfg::block_L_kv/Cfg::mma_k][Cfg::warp_L_q/Cfg::mma_m][4];
  uint32_t v_frag[Cfg::block_L_kv/Cfg::mma_k][Cfg::D/Cfg::mma_n][4]; 
  float o_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::D/Cfg::mma_n][8] = {0.0}; 
  float m_i[Cfg::warp_L_q/Cfg::mma_m][2]; 
  float l_i[Cfg::warp_L_q/Cfg::mma_m][2]; 
  const float scale = rsqrtf((float)Cfg::D);
  
  #pragma unroll
  for (int m=0; m<Cfg::warp_L_q/Cfg::mma_m; m++)
  {
      m_i[m][0] = -INFINITY;
      m_i[m][1] = -INFINITY;

      l_i[m][0] = 0.f;
      l_i[m][1] = 0.f;
  }

  if (w == 0)
  {   
    if (t == 0)
    { 
      mbarrier_arrive_expect_tx(q_bar, Cfg::Qs_bytes);
      cp_async_bulk_tensor_3d(Qs,&q_map,0,block_start_lq,block_start_BH,q_bar);

    }
    else mbarrier_arrive(q_bar);
  }
  __syncthreads();

  mbarrier_wait_parity(q_bar,0); 

  for (int m = 0; m< Cfg::warp_L_q/Cfg::mma_m; m++)
  {
    for (int k = 0; k < Cfg::D/Cfg::mma_k; k++)
    {
      uint32_t q_frag_ld_addr = Qs + (((warp_start_lq + (m*Cfg::mma_m) + (l%16))*Cfg::D + ((k*Cfg::mma_k + (8*(l/16)))))*sizeof(nv_bfloat16));
      wa::ldmatrix_m8n8_x4_b16(q_frag[k][m],q_frag_ld_addr);
    }
  }
  __syncthreads(); 
  
  auto TMA_LOAD_K = [&](int blkv)
  {
    if (w == 0)
    {
      if (l == 0)
      {
        mbarrier_arrive_expect_tx(k_bar, Cfg::Ks_bytes);
        cp_async_bulk_tensor_3d(Ks, &k_map, blkv*Cfg::block_L_kv, 0, block_start_BH, k_bar);
      }
      else mbarrier_arrive(k_bar);
    }
    __syncthreads();
  };

  auto TMA_LOAD_V = [&](int blkv) 
  {
    if (w == 0)
    {
      if (l == 0)
      {
        mbarrier_arrive_expect_tx(v_bar, Cfg::Vs_bytes);
        cp_async_bulk_tensor_3d(Vs, &v_map, blkv*Cfg::block_L_kv, 0, block_start_BH, v_bar);
      }
      else mbarrier_arrive(v_bar);

    }
    __syncthreads();
  };
  auto K_smem_rmem = [&](uint32_t kt_frag[Cfg::D/Cfg::mma_k][Cfg::block_L_kv/Cfg::mma_n][4])
  {
    for (int k = 0; k < Cfg::D/Cfg::mma_k; k++)
    {
      for (int n = 0; n < Cfg::block_L_kv/Cfg::mma_n; n++)
      {
        uint32_t kt_ld_addr = Ks + compact_swizzle<Cfg::swizzle_num>(((k*Cfg::mma_k + (l%16))*Cfg::block_L_kv + (n*Cfg::mma_n + (8*(l/16))))*sizeof(nv_bfloat16));
        wa::ldmatrix_m8n8_x4_trans_b16(kt_frag[k][n], kt_ld_addr);
      }
    }
  };

  auto V_smem_rmem = [&](uint32_t v_frag[Cfg::block_L_kv/Cfg::mma_k][Cfg::D/Cfg::mma_n][4])
  {
    for (int k = 0; k < Cfg::block_L_kv/Cfg::mma_k;k++)
    {
      for (int n = 0; n < Cfg::D/Cfg::mma_n; n++)
      {
        uint32_t v_ld_addr = Vs + compact_swizzle<Cfg::swizzle_num>(((n*Cfg::mma_n + ((l%8)+(8*(l/16))))*Cfg::block_L_kv + (k*Cfg::mma_k + (8*((l/8)%2))))*sizeof(nv_bfloat16));
        wa::ldmatrix_m8n8_x4_b16(v_frag[k][n],v_ld_addr);
      }
    }
  };

  auto mma_Q_KT_scale = [&](float s_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::block_L_kv/Cfg::mma_n][8],
                            uint32_t q_frag[Cfg::D/Cfg::mma_k][Cfg::warp_L_q/Cfg::mma_m][4],
                            uint32_t kt_frag[Cfg::D/Cfg::mma_k][Cfg::block_L_kv/Cfg::mma_n][4])
  {
    for (int k = 0; k < Cfg::D/Cfg::mma_k; k++)
    {
      for (int m = 0; m < Cfg::warp_L_q/Cfg::mma_m; m++)
      {
        for(int n = 0; n < Cfg::block_L_kv/Cfg::mma_n; n++)
          wa::mma_m16n16k16_row_col_f32_bf16(s_frag[m][n], q_frag[k][m], kt_frag[k][n]);
      }
    }
    __syncthreads();
    for (int m = 0; m < Cfg::warp_L_q/Cfg::mma_m; m++)
    {
      for (int n = 0; n < Cfg::block_L_kv/Cfg::mma_n; n++)
      {
        float* s = s_frag[m][n];

        s[0] *= scale;
        s[1] *= scale;
        s[2] *= scale;
        s[3] *= scale;
        s[4] *= scale;
        s[5] *= scale;
        s[6] *= scale;
        s[7] *= scale;
      }
    }
  };

  auto mma_P_V = [&](float o_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::D/Cfg::mma_n][8], 
                    uint32_t p_frag_packed[Cfg::block_L_kv/Cfg::mma_k][Cfg::warp_L_q/Cfg::mma_m][4],
                    uint32_t v_frag[Cfg::block_L_kv/Cfg::mma_k][Cfg::D/Cfg::mma_n][4])
  
  {
    for (int k = 0; k < Cfg::block_L_kv/Cfg::mma_k; k++)
    {
      for (int m = 0; m < Cfg::warp_L_q/Cfg::mma_m; m++)
      {
        for (int n = 0; n < Cfg::D/Cfg::mma_n; n++)
        {
          wa::mma_m16n16k16_row_col_f32_bf16(
          o_frag[m][n],
          p_frag_packed[k][m],
          v_frag[k][n]);
        }
      }
    }
  };

  int parity = 0; 

  //prologue 
  TMA_LOAD_K(0); 
  //1TMA in flight
  
  for (int blkv = 0; blkv < (Cfg::L_kv/Cfg::block_L_kv) - 1; blkv++)
  {
    float s_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::block_L_kv/Cfg::mma_n][8] = {0.0}; 
    __syncthreads();
    TMA_LOAD_V(blkv);//2TMAs in flight 
    mbarrier_wait_parity(k_bar, parity); //1 TMA in flight
    K_smem_rmem(kt_frag);
    __syncthreads(); //wait for kt_frag to get into rmem  
    TMA_LOAD_K(blkv+1); //2 TMAs in flight 
    mma_Q_KT_scale(s_frag, q_frag, kt_frag); // one set of MMAs in flight
    __syncthreads();
    flash_softmax_update<Cfg>(s_frag, o_frag, m_i, l_i); 
    pack_p_frag<Cfg>(s_frag, p_frag_packed);
    __syncthreads();
    mbarrier_wait_parity(v_bar, parity); // 1 TMA in flight 
    V_smem_rmem(v_frag);
    mma_P_V(o_frag,p_frag_packed,v_frag);
    parity ^= 1; 
  }
  int blkv = (Cfg::L_kv/Cfg::block_L_kv) - 1;
  float s_frag[Cfg::warp_L_q/Cfg::mma_m][Cfg::block_L_kv/Cfg::mma_n][8] = {0.0}; 
  __syncthreads();
  TMA_LOAD_V(blkv);//2TMAs in flight 
  mbarrier_wait_parity(k_bar, parity); //1 TMA in flight
  K_smem_rmem(kt_frag);
  __syncthreads(); //wait for kt_frag to get into rmem  
  mma_Q_KT_scale(s_frag, q_frag, kt_frag); // one set of MMAs in flight
  __syncthreads();
  flash_softmax_update<Cfg>(s_frag, o_frag, m_i, l_i); 
  pack_p_frag<Cfg>(s_frag, p_frag_packed);
  __syncthreads();
  mbarrier_wait_parity(v_bar, parity); // 0 TMA in flight 
  V_smem_rmem(v_frag);
  mma_P_V(o_frag,p_frag_packed,v_frag);

  __syncthreads();

  float2 *O2 = reinterpret_cast<float2*>(O); 
  int ldO2 = Cfg::D/2; 
  int head_base = block_start_BH * Cfg::L_q * ldO2;

  for (int m = 0; m < Cfg::warp_L_q/Cfg::mma_m; m++)
  {
    for (int n = 0; n < Cfg::D/Cfg::mma_n; n++)
    {
      int O_row = block_start_lq + warp_start_lq + (m*Cfg::mma_m) + (l/4); 
      int O_col = (n*Cfg::mma_n + (2*(l%4)))/2; 
      float inv_l0 = 1.f / l_i[m][0];
      float inv_l1 = 1.f / l_i[m][1];

      float2 v00 = {o_frag[m][n][0] * inv_l0, o_frag[m][n][1] * inv_l0};
      float2 v01 = {o_frag[m][n][4] * inv_l0, o_frag[m][n][5] * inv_l0};
      float2 v10 = {o_frag[m][n][2] * inv_l1, o_frag[m][n][3] * inv_l1};
      float2 v11 = {o_frag[m][n][6] * inv_l1, o_frag[m][n][7] * inv_l1};

      O2[head_base + O_row*ldO2 + O_col] = v00; 
      O2[head_base + O_row*ldO2 + (O_col+4)] = v01; 
      O2[head_base + (O_row+8)*ldO2 + O_col] = v10; 
      O2[head_base + (O_row+8)*ldO2 + (O_col+4)] = v11; 
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
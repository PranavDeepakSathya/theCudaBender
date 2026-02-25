#pragma once
#include "../atoms/all.cuh"




template <class Cfg>
__global__ void attention_kernel(__grid_constant__ const CUtensorMap q_map,
                                 __grid_constant__ const CUtensorMap k_map,
                                __grid_constant__ const CUtensorMap v_map,
                                float*O)


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
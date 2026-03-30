// bench.cu — no torch, no pybind. Pure CUDA + C-ABI exports.
#include "../atoms/all.cuh"
#include "config.cuh"
#include "kernel.cuh"

using Cfg = AttnConfig;

extern "C" {

void build_tma_maps(
    nv_bfloat16* Q,
    nv_bfloat16* K,
    nv_bfloat16* V,
    CUtensorMap* q_map_out,
    CUtensorMap* k_map_out,
    CUtensorMap* v_map_out)
{
    *q_map_out = TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
        Q,
        {(uint64_t)Cfg::BH, (uint64_t)2*Cfg::L_q,  (uint64_t)Cfg::D/2},
        {1u, (uint32_t)2*Cfg::block_L_q, (uint32_t)Cfg::D/2},
        {2, 1, 0},
        Cfg::D_swizzle_mode
    );
    *k_map_out = TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
        K,
        {(uint64_t)Cfg::BH, (uint64_t)2*Cfg::L_kv,  (uint64_t)Cfg::D/2},
        {1u, (uint32_t)2*Cfg::block_L_kv, (uint32_t)Cfg::D/2},
        {2, 1, 0},
        Cfg::D_swizzle_mode
    );
    *v_map_out = TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
        V,
        {(uint64_t)Cfg::BH, (uint64_t)Cfg::L_kv, (uint64_t)Cfg::D},
        {1u, (uint32_t)Cfg::block_L_kv, (uint32_t)Cfg::D},
        {1, 2, 0},
        Cfg::swizzle_mode
    );
}

int init_kernel()
{
    cudaGetLastError();
    cudaFuncSetAttribute(
        attention_kernel<Cfg>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    cudaFuncSetAttribute(
        attention_kernel<Cfg>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        Cfg::shared_bytes
    );
    return (int)cudaGetLastError();
}

int run_attention(
    CUtensorMap* q_map,
    CUtensorMap* k_map,
    CUtensorMap* v_map,
    float*       O)
{
    cudaLaunchConfig_t config = {};
    config.gridDim            = Cfg::grid_size;
    config.blockDim           = Cfg::block_size;
    config.dynamicSmemBytes   = Cfg::shared_bytes;
    config.stream             = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id             = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {1, 1, 1};
    config.attrs    = attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, attention_kernel<Cfg>, *q_map, *k_map, *v_map, O);
    return (int)cudaGetLastError();
}

} // extern "C"

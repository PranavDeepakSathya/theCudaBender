// bench.cu — no torch, no pybind. Pure CUDA + C-ABI exports.
#include "../atoms/all.cuh"
#include "config.cuh"
#include "kernel.cuh"
#include "smem_allocator.cuh"

using Cfg = GemmConfig;

extern "C" {

void build_tma_maps(
    nv_bfloat16* A,
    nv_bfloat16* B,
    CUtensorMap* a_map_out,
    CUtensorMap* b_map_out
)
{
    *a_map_out = TmaDescriptor<nv_bfloat16>::create_2d_row_major(
        A,
        {Cfg::M, Cfg::K},
        {Cfg::BM, Cfg::BK},
        Cfg::swizzle_mode
    );
    *b_map_out = TmaDescriptor<nv_bfloat16>::create_2d_col_major(
        B,
        {Cfg::K, Cfg::N},
        {Cfg::BK, Cfg::BN},
        Cfg::swizzle_mode
    );
}

// call once before benching — returns 0 on success
int init_kernel()
{
    cudaGetLastError();
    cudaFuncSetAttribute(
        matmul_kernel<Cfg>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    cudaFuncSetAttribute(
        matmul_kernel<Cfg>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        Cfg::shared_bytes
    );
    return (int)cudaGetLastError();
}

int run_gemm(
    nv_bfloat16* A,
    nv_bfloat16* B,
    float*       C,
    CUtensorMap* a_map,
    CUtensorMap* b_map
)
{
    cudaLaunchConfig_t config = {};
    config.gridDim          = Cfg::grid_size;
    config.blockDim         = Cfg::block_size;
    config.dynamicSmemBytes = Cfg::shared_bytes;
    config.stream           = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id             = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {1, 1, 1};
    config.attrs    = attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, matmul_kernel<Cfg>, *a_map, *b_map, C);
    return (int)cudaGetLastError();
}

} // extern "C"

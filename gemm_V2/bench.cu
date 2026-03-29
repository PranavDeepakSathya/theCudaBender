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
    *a_map_out = TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
        A,
        std::array<uint64_t,3>{(uint64_t)Cfg::L, (uint64_t)Cfg::M, (uint64_t)Cfg::K},
        std::array<uint32_t,3>{1u, (uint32_t)Cfg::BM, (uint32_t)Cfg::BK},
        std::array<int,3>{2, 1, 0},
        Cfg::swizzle_mode
    );
    *b_map_out = TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
        B,
        std::array<uint64_t,3>{(uint64_t)Cfg::L, (uint64_t)Cfg::K, (uint64_t)Cfg::N},
        std::array<uint32_t,3>{1u, (uint32_t)Cfg::BK, (uint32_t)Cfg::BN},
        std::array<int,3>{1, 2, 0},
        Cfg::swizzle_mode
    );
}

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
    float*       bias,
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

    cudaLaunchKernelEx(&config, matmul_kernel<Cfg>, *a_map, *b_map, C, bias);
    return (int)cudaGetLastError();
}

} // extern "C"

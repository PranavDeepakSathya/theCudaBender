#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.cuh"
#include "kernel.cuh"
#include "../atoms/all.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

using Cfg = GemmConfig;

// ------------------------------------------------------------
// profiler constants
// ------------------------------------------------------------

constexpr int THREADS_PER_BLOCK =
    Cfg::block_size;

constexpr int NUM_STREAMS =
    Cfg::grid_size * THREADS_PER_BLOCK;

constexpr int NUM_PROF_EVENTS = 20000;

// ------------------------------------------------------------
// GEMM entrypoint
// ------------------------------------------------------------

torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");

    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");

    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");

    TORCH_CHECK(
        B.stride(0) == 1,
        "B must be column-major (stride0=1).\n"
        "Construct with:\n"
        "  B = torch.randn((N,K), device='cuda', dtype=torch.bfloat16).t()\n"
    );

    TORCH_CHECK(A.size(0) == Cfg::M, "A.shape mismatch");
    TORCH_CHECK(A.size(1) == Cfg::K, "A.shape mismatch");

    TORCH_CHECK(B.size(0) == Cfg::K, "B.shape mismatch");
    TORCH_CHECK(B.size(1) == Cfg::N, "B.shape mismatch");

    // ------------------------------------------------------------
    // Allocate output
    // ------------------------------------------------------------

    auto C = torch::empty(
        {Cfg::M, Cfg::N},
        torch::TensorOptions()
            .device(A.device())
            .dtype(torch::kFloat32)
    );

    // ------------------------------------------------------------
    // Allocate profiler buffer
    // ------------------------------------------------------------

    auto profiler = torch::zeros(
        {NUM_STREAMS, 1 + NUM_PROF_EVENTS * 4},
        torch::TensorOptions()
            .device(A.device())
            .dtype(torch::kInt64)
    );

    int64_t* profiler_ptr = profiler.data_ptr<int64_t>();

    // ------------------------------------------------------------
    // Tensor pointers
    // ------------------------------------------------------------

    nv_bfloat16* A_ptr =
        reinterpret_cast<nv_bfloat16*>(A.data_ptr<at::BFloat16>());

    nv_bfloat16* B_ptr =
        reinterpret_cast<nv_bfloat16*>(B.data_ptr<at::BFloat16>());

    float* C_ptr = C.data_ptr<float>();

    // ------------------------------------------------------------
    // Build TMA tensor maps
    // ------------------------------------------------------------

    CUtensorMap a_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            A_ptr,
            {Cfg::M, Cfg::K},
            {Cfg::BM, Cfg::BK},
            Cfg::swizzle_mode
        );

    CUtensorMap b_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            B_ptr,
            {Cfg::K, Cfg::N},
            {Cfg::BK, Cfg::BN},
            Cfg::swizzle_mode
        );

    // ------------------------------------------------------------
    // Launch kernel
    // ------------------------------------------------------------

    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    launch_matmul<Cfg>(
        launcher,
        a_map,
        b_map,
        C_ptr,
        profiler_ptr,
        NUM_PROF_EVENTS
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // return profiler buffer so Python can build trace
    return profiler;
}

// ------------------------------------------------------------
// Config introspection (for autotune JSON)
// ------------------------------------------------------------

std::vector<int64_t> config_signature()
{
    return {
        Cfg::acc_per_warp_m,
        Cfg::acc_per_warp_n,
        Cfg::warp_k_iters,
        Cfg::warps_per_block_m,
        Cfg::warps_per_block_n,
        Cfg::shared_bytes
    };
}

// ------------------------------------------------------------
// Shape export
// ------------------------------------------------------------

std::vector<int64_t> shape()
{
    return {Cfg::M, Cfg::N, Cfg::K};
}

// ------------------------------------------------------------
// PyBind
// ------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &gemm, "SM120 BF16 GEMM (fp32 accumulate)");
    m.def("shape", &shape);
    m.def("config_signature", &config_signature);
}
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
// GEMM entrypoint
// ------------------------------------------------------------
torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be fp16");

    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");

    // ---- HARD REQUIRE: B must be col-major ----
    TORCH_CHECK(
        B.stride(0) == 1,
        "B must be column-major (stride0=1).\n"
        "Construct with:\n"
        "  B = torch.randn((N,K), device='cuda', dtype=torch.float16).t()\n"
    );

    TORCH_CHECK(A.size(0) == Cfg::M, "A.shape mismatch");
    TORCH_CHECK(A.size(1) == Cfg::K, "A.shape mismatch");

    TORCH_CHECK(B.size(0) == Cfg::K, "B.shape mismatch");
    TORCH_CHECK(B.size(1) == Cfg::N, "B.shape mismatch");

    // ------------------------------------------------------------
    // Allocate output (FP16, row-major)
    // ------------------------------------------------------------
    auto C = torch::empty(
        {Cfg::M, Cfg::N},
        torch::TensorOptions()
            .device(A.device())
            .dtype(torch::kFloat16)
    );

    __half* A_ptr =
        reinterpret_cast<__half*>(A.data_ptr<at::Half>());

    __half* B_ptr =
        reinterpret_cast<__half*>(B.data_ptr<at::Half>());

    __half* C_ptr =
        reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    // ------------------------------------------------------------
    // Build TMA tensor maps
    // ------------------------------------------------------------

    CUtensorMap a_map =
        TmaDescriptor<__half>::create_2d_row_major(
            A_ptr,
            {Cfg::M, Cfg::K},
            {Cfg::BM, Cfg::BK},
            Cfg::ab_swizzle_mode
        );

    CUtensorMap b_map =
        TmaDescriptor<__half>::create_2d_col_major(
            B_ptr,
            {Cfg::K, Cfg::N},
            {Cfg::BK, Cfg::BN},
            Cfg::ab_swizzle_mode
        );

    // ---- C is row-major ----
    CUtensorMap c_map =
        TmaDescriptor<__half>::create_2d_row_major(
            C_ptr,
            {Cfg::M, Cfg::N},
            {Cfg::BM, Cfg::BN},
            Cfg::c_swizzle_mode
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
        c_map
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

// ------------------------------------------------------------
// Config introspection
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
    m.def("gemm", &gemm, "SM120 FP16 GEMM (fp16 accumulate)");
    m.def("shape", &shape);
    m.def("config_signature", &config_signature);
}
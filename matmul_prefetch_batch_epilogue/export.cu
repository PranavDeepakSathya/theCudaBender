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

    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");

    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");

    // ---- HARD REQUIRE: B must be col-major ----
    TORCH_CHECK(
        B.stride(1) == 1,
        "B must be column-major (stride0=1).\n"
        "Construct with:\n"
        "  B = torch.randn((B,N,K), device='cuda', dtype=torch.bfloat16).t()\n"
    );
    TORCH_CHECK(A.size(0) == Cfg::L, "A.shape mismatch")
    TORCH_CHECK(A.size(1) == Cfg::M, "A.shape mismatch");
    TORCH_CHECK(A.size(2) == Cfg::K, "A.shape mismatch");

    TORCH_CHECK(B.size(0) == Cfg::L, "B.shape mismatch");
    TORCH_CHECK(B.size(1) == Cfg::K, "B.shape mismatch");
    TORCH_CHECK(B.size(2) == Cfg::N, "B.shape mismatch");

    // ------------------------------------------------------------
    // Allocate output
    // ------------------------------------------------------------
    auto C = torch::empty(
        {Cfg::L, Cfg::M, Cfg::N},
        torch::TensorOptions()
            .device(A.device())
            .dtype(torch::kFloat32)
    );

    nv_bfloat16* A_ptr =
        reinterpret_cast<nv_bfloat16*>(A.data_ptr<at::BFloat16>());

    nv_bfloat16* B_ptr =
        reinterpret_cast<nv_bfloat16*>(B.data_ptr<at::BFloat16>());

    float* C_ptr = C.data_ptr<float>();

    // ------------------------------------------------------------
    // Build TMA tensor maps
    // ------------------------------------------------------------
    CUtensorMap a_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            A_ptr,
            {Cfg::L,Cfg::M, Cfg::K},
            {1,Cfg::BM, Cfg::BK},
            {2,1,0},
            Cfg::swizzle_mode
        );

    CUtensorMap b_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            B_ptr,
            {Cfg::L, Cfg::K, Cfg::N},
            {1, Cfg::BK, Cfg::BN},
            {1,2,0},
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

    launch_matmul<Cfg>(launcher, a_map, b_map, C_ptr);

    // ---- REQUIRED for autotuning ----
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
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
    return {Cfg::L, Cfg::M, Cfg::N, Cfg::K};
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

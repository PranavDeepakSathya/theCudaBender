#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.cuh"
#include "kernel.cuh"
#include "../atoms/all.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

using Cfg = AttnConfig;

torch::Tensor attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bf16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bf16");

    TORCH_CHECK(Q.dim() == 2, "Q must be 2D");
    TORCH_CHECK(K.dim() == 2, "K must be 2D");
    TORCH_CHECK(V.dim() == 2, "V must be 2D");

    TORCH_CHECK(Q.is_contiguous(), "Q must be row-major");
    TORCH_CHECK(K.stride(0) == 1, "K must be column-major");
    TORCH_CHECK(V.is_contiguous(), "V must be row-major");

    TORCH_CHECK(Q.size(0) == Cfg::LQ, "Q shape mismatch");
    TORCH_CHECK(Q.size(1) == Cfg::D,  "Q shape mismatch");

    TORCH_CHECK(K.size(0) == Cfg::D,  "K shape mismatch");
    TORCH_CHECK(K.size(1) == Cfg::LK, "K shape mismatch");

    TORCH_CHECK(V.size(0) == Cfg::LK, "V shape mismatch");
    TORCH_CHECK(V.size(1) == Cfg::D,  "V shape mismatch");

    auto O = torch::empty(
        {Cfg::LQ, Cfg::D},
        torch::TensorOptions()
            .device(Q.device())
            .dtype(torch::kFloat32)
    );

    nv_bfloat16* Q_ptr =
        reinterpret_cast<nv_bfloat16*>(Q.data_ptr<at::BFloat16>());
    nv_bfloat16* K_ptr =
        reinterpret_cast<nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    nv_bfloat16* V_ptr =
        reinterpret_cast<nv_bfloat16*>(V.data_ptr<at::BFloat16>());

    float* O_ptr = O.data_ptr<float>();

    CUtensorMap q_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            Q_ptr,
            {Cfg::LQ, Cfg::D},
            {Cfg::LQ, Cfg::D}
        );

    CUtensorMap k_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            K_ptr,
            {Cfg::D, Cfg::LK},
            {Cfg::D, Cfg::LK}
        );

    CUtensorMap v_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            V_ptr,
            {Cfg::LK, Cfg::D},
            {Cfg::LK, Cfg::D}
        );

    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    launch_attention<Cfg>(launcher, q_map, k_map, v_map, O_ptr);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;
}

std::vector<int64_t> shape()
{
    return {Cfg::LQ, Cfg::D, Cfg::LK};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "Single-warp Attention");
    m.def("shape", &shape);
}
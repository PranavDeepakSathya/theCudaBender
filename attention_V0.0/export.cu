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

    // Layout contracts
    TORCH_CHECK(Q.is_contiguous(), "Q must be row-major");
    TORCH_CHECK(K.is_contiguous(), "K must be row-major");
    TORCH_CHECK(V.stride(0) == 1, "V must be column-major");

    // Shape contracts
    TORCH_CHECK(Q.size(0) == Cfg::L_q, "Q shape mismatch");
    TORCH_CHECK(Q.size(1) == Cfg::D,  "Q shape mismatch");

    TORCH_CHECK(K.size(0) == Cfg::L_kv, "K shape mismatch");
    TORCH_CHECK(K.size(1) == Cfg::D,  "K shape mismatch");

    TORCH_CHECK(V.size(0) == Cfg::L_kv, "V shape mismatch");
    TORCH_CHECK(V.size(1) == Cfg::D,  "V shape mismatch");

    auto O = torch::empty(
        {Cfg::L_q, Cfg::D},
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

    // ---- TMA Descriptors ----

    CUtensorMap q_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            Q_ptr,
            {Cfg::L_q, Cfg::D},
            {Cfg::block_L_q, Cfg::D}
        );

    CUtensorMap k_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            K_ptr,
            {Cfg::L_kv, Cfg::D},
            {Cfg::block_L_kv, Cfg::D}
        );

    CUtensorMap v_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            V_ptr,
            {Cfg::L_kv, Cfg::D},
            {Cfg::block_L_kv, Cfg::D}
        );

    // ---- Launch ----

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
  return {Cfg::L_q, Cfg::D, Cfg::L_kv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "2d-attention");
    m.def("shape", &shape);
}
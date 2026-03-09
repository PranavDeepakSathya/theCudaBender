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

    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    TORCH_CHECK(K.dtype() == torch::kBFloat16);
    TORCH_CHECK(V.dtype() == torch::kBFloat16);

    TORCH_CHECK(Q.dim() == 3);
    TORCH_CHECK(K.dim() == 3);
    TORCH_CHECK(V.dim() == 3);

    TORCH_CHECK(Q.is_contiguous());
    TORCH_CHECK(K.is_contiguous());

    TORCH_CHECK(Q.size(0) == Cfg::BH);
    TORCH_CHECK(Q.size(1) == Cfg::L_q);
    TORCH_CHECK(Q.size(2) == Cfg::D);

    TORCH_CHECK(K.size(0) == Cfg::BH);
    TORCH_CHECK(K.size(1) == Cfg::L_kv);
    TORCH_CHECK(K.size(2) == Cfg::D);

    TORCH_CHECK(V.size(0) == Cfg::BH);
    TORCH_CHECK(V.size(1) == Cfg::L_kv);
    TORCH_CHECK(V.size(2) == Cfg::D);

    auto O = torch::empty(
        {Cfg::BH, Cfg::L_q, Cfg::D},
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


    //--------------------------------
    // TMA descriptors (generic layout)
    //--------------------------------

    CUtensorMap q_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            Q_ptr,
            {Cfg::BH, Cfg::L_q, Cfg::D},
            {1, Cfg::block_L_q, Cfg::D},
            {2,1,0}   // D,L,BH  (row-major inner)
        );

    CUtensorMap k_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            K_ptr,
            {Cfg::BH, 2*Cfg::L_kv, Cfg::D/2},
            {1, 2*Cfg::block_L_kv, Cfg::D/2},
            {2,1,0}   // D,L,BH
        );

    CUtensorMap v_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            V_ptr,
            {Cfg::BH, Cfg::L_kv, Cfg::D},
            {1, Cfg::block_L_kv, Cfg::D},
            {1,2,0}   // L,D,BH  (column-major inner)
        );


    //--------------------------------
    // Launch
    //--------------------------------

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
    return {Cfg::BH, Cfg::L_q, Cfg::D, Cfg::L_kv};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "3d-attention");
    m.def("shape", &shape);
}
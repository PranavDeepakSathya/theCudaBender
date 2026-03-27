#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.cuh"
#include "kernel.cuh"
#include "../atoms/all.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

using Cfg = AttnConfig;

// only profiling block 0 -> num_warps streams
constexpr int NUM_PROF_STREAMS = Cfg::num_warps;
constexpr int NUM_PROF_EVENTS  = 20000;

std::vector<torch::Tensor> attention(
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

    auto profiler = torch::zeros(
        {NUM_PROF_STREAMS, 1 + NUM_PROF_EVENTS * 4},
        torch::TensorOptions()
            .device(Q.device())
            .dtype(torch::kInt64)
    );

    nv_bfloat16* Q_ptr =
        reinterpret_cast<nv_bfloat16*>(Q.data_ptr<at::BFloat16>());

    nv_bfloat16* K_ptr =
        reinterpret_cast<nv_bfloat16*>(K.data_ptr<at::BFloat16>());

    nv_bfloat16* V_ptr =
        reinterpret_cast<nv_bfloat16*>(V.data_ptr<at::BFloat16>());

    float* O_ptr = O.data_ptr<float>();
    int64_t* prof_ptr = profiler.data_ptr<int64_t>();

    //--------------------------------
    // TMA descriptors (generic layout)
    //--------------------------------

    CUtensorMap q_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            Q_ptr,
            {Cfg::BH, Cfg::L_q, Cfg::D},
            {1, Cfg::block_L_q, Cfg::D},
            {2,1,0} // D,L,BH  (row-major inner)

        );

    CUtensorMap k_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            K_ptr,
            {Cfg::BH, 2*Cfg::L_kv, Cfg::D/2},
            {1, 2*Cfg::block_L_kv, Cfg::D/2},
            {2,1,0},
            Cfg::D_swizzle_mode

        );

    CUtensorMap v_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            V_ptr,
            {Cfg::BH, Cfg::L_kv, Cfg::D},
            {1, Cfg::block_L_kv, Cfg::D},
            {1,2,0},
            Cfg::swizzle_mode
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

    launch_attention<Cfg>(launcher, q_map, k_map, v_map, O_ptr, prof_ptr, NUM_PROF_EVENTS);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O, profiler};
}


std::vector<int64_t> shape()
{
    return {Cfg::BH, Cfg::L_q, Cfg::D, Cfg::L_kv};
}

int64_t num_warps()
{
    return Cfg::num_warps;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "3d-attention with profiler");
    m.def("shape", &shape);
    m.def("num_warps", &num_warps);
}

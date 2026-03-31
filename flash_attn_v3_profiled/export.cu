#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.cuh"
#include "kernel.cuh"
#include "../atoms/all.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <intra_kernel_profiler/trace/trace.cuh>

using Cfg = AttnConfig;
namespace ikp = intra_kernel_profiler::trace;

static ikp::HostSession g_ikp_session;
static bool g_ikp_initialized = false;

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
            {2,1,0} // D,L,BH  D Major

        );

    CUtensorMap k_map =
        TmaDescriptor<nv_bfloat16>::create_with_layout<3>(
            K_ptr,
            {Cfg::BH, Cfg::L_kv, Cfg::D},
            {1, Cfg::block_L_kv, Cfg::D},
            {1,2,0}, //Lkv major
            Cfg::swizzle_mode
  

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
    // IKP trace setup
    //--------------------------------
    if (!g_ikp_initialized) {
        g_ikp_session.set_region_names({
            "_unused",       // id=0 (not used)
            "total",         // id=1
            "load_q",        // id=2
            "load_k_gmem",   // id=3
            "load_k_smem",   // id=4
            "compute_qk",    // id=5
            "softmax",       // id=6
            "load_v_gmem",   // id=7
            "load_v_smem",   // id=8
            "compute_pv"     // id=9
        });
        g_ikp_session.init(32768, Cfg::grid_size, Cfg::block_size);
        g_ikp_initialized = true;
    }
    g_ikp_session.reset();

    //--------------------------------
    // Launch
    //--------------------------------

    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    launch_attention<Cfg>(launcher, q_map, k_map, v_map, O_ptr,
                          g_ikp_session.global_buffer());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;
}


std::vector<int64_t> shape()
{
    return {Cfg::BH, Cfg::L_q, Cfg::D, Cfg::L_kv};
}

void dump_trace(const std::string& path)
{
    cudaDeviceSynchronize();
    ikp::TraceWriteOptions opt;
    opt.scale = 1.0;  // raw ns
    opt.emit_summary_json = true;
    opt.summary_dump_by_block_warp = true;
    g_ikp_session.write_trace(path, opt);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "3d-attention");
    m.def("shape", &shape);
    m.def("dump_trace", &dump_trace, "Dump IKP trace to JSON file");
}
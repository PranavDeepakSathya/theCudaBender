// k0/export.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "config.cuh"
#include "kernel.cuh"

using Cfg = GemmConfig;

torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A, B must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16, "A, B must be bf16");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");
  TORCH_CHECK(B.stride(0) == 1,
    "B must be column-major (stride0=1).\n"
    "Construct with: B = torch.randn((N,K), device='cuda', dtype=torch.bfloat16).t()\n");

  TORCH_CHECK(A.size(0) == Cfg::M && A.size(1) == Cfg::K, "A shape mismatch");
  TORCH_CHECK(B.size(0) == Cfg::K && B.size(1) == Cfg::N, "B shape mismatch");

  auto C = torch::empty(
    {Cfg::M, Cfg::N},
    torch::TensorOptions().device(A.device()).dtype(torch::kFloat32)
  );

  nv_bfloat16* A_ptr = reinterpret_cast<nv_bfloat16*>(A.data_ptr<at::BFloat16>());
  nv_bfloat16* B_ptr = reinterpret_cast<nv_bfloat16*>(B.data_ptr<at::BFloat16>());
  float*        C_ptr = C.data_ptr<float>();

  CUtensorMap a_map =
          TmaDescriptor<nv_bfloat16>::create_with_layout<2>(
              A_ptr,
             {(uint64_t)Cfg::M, (uint64_t)Cfg::K},
              {(uint32_t)Cfg::BM, (uint32_t)Cfg::BK},
              {1, 0},
              Cfg::swizzle_mode,
              CU_TENSOR_MAP_INTERLEAVE_NONE,
              CU_TENSOR_MAP_L2_PROMOTION_NONE,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
          );

      CUtensorMap b_map =
          TmaDescriptor<nv_bfloat16>::create_with_layout<2>(
              B_ptr,
              {(uint64_t)Cfg::K, (uint64_t)Cfg::N},
              {(uint32_t)Cfg::BK, (uint32_t)Cfg::BN},
              {0,1},
              Cfg::swizzle_mode,
              CU_TENSOR_MAP_INTERLEAVE_NONE,
              CU_TENSOR_MAP_L2_PROMOTION_NONE,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
          );

  NaiveLauncher launcher(Cfg::grid_size, 1, Cfg::block_size, Cfg::shared_bytes);
  launch_matmul<Cfg>(launcher, a_map, b_map, C_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return C;
}

std::vector<int64_t> shape() { return {Cfg::M, Cfg::N, Cfg::K}; }

std::vector<int64_t> config_signature()
{
  return {
    Cfg::acc_per_warp_m, Cfg::acc_per_warp_n,
    Cfg::warps_per_block_m, Cfg::warps_per_block_n,
    Cfg::BK, Cfg::shared_bytes
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gemm",             &gemm);
  m.def("shape",            &shape);
  m.def("config_signature", &config_signature);
}
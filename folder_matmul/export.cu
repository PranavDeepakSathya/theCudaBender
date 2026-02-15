
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.cuh"
#include "kernel.cuh"
#include "../atoms/all.cuh"

using Cfg = GemmConfig;


torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA");

  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");


  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");

  TORCH_CHECK(A.size(0) == Cfg::M, "A.shape mismatch");
  TORCH_CHECK(A.size(1) == Cfg::K, "A.shape mismatch");

  TORCH_CHECK(B.size(0) == Cfg::K, "B.shape mismatch");
  TORCH_CHECK(B.size(1) == Cfg::N, "B.shape mismatch");

  auto C = torch::empty({Cfg::M, Cfg::N},
                        torch::TensorOptions()
                            .device(A.device())
                            .dtype(torch::kFloat32));

  nv_bfloat16* A_ptr = (nv_bfloat16*)A.data_ptr<at::BFloat16>();
  nv_bfloat16* B_ptr = (nv_bfloat16*)B.data_ptr<at::BFloat16>();
  float*       C_ptr = C.data_ptr<float>();


  CUtensorMap a_map =
      TmaDescriptor<nv_bfloat16>::create_2d_row_major(
          A_ptr,
          {Cfg::M, Cfg::K},
          {Cfg::BM, Cfg::BK}
          /*swizzle=*/
      );

  CUtensorMap b_map =
      TmaDescriptor<nv_bfloat16>::create_2d_col_major(
          B_ptr,
          {Cfg::K, Cfg::N},
          {Cfg::BK, Cfg::BN}
          /*swizzle=*/
      );


  NaiveLauncher launcher(
      Cfg::grid_size,
      1,
      Cfg::block_size,
      Cfg::shared_bytes
  );

  launch_matmul<Cfg>(launcher, a_map, b_map, C_ptr);

  return C;
}

std::vector<int64_t> get_shape() {
    return {GemmConfig::M, GemmConfig::N, GemmConfig::K};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gemm", &gemm, "SM120 BF16 GEMM (fp32 accumulate)");
  m.def("shape", &get_shape);
}

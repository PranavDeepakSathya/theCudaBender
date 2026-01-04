#include "atoms/utils.cuh"
#include "atoms/naive_launcher.cuh"
#include "atoms/tma_factory.cuh"
#include "atoms/naive_tensor.cuh"

constexpr int wmma_m = 16; 
constexpr int wmma_k = 16; 
constexpr int wmma_n = 8; 
constexpr int n_consumer_warps = 16; 
constexpr int warp_shape_n = 4; 
constexpr int warp_shape_m = n_consumer_warps/warp_shape_n; 
constexpr int k_iter = 4;
constexpr int m = wmma_m*warp_shape_m;
constexpr int n = wmma_n*warp_shape_n;
constexpr int k = k_iter*wmma_k;

constexpr int n_tpb = n_consumer_warps*32; 
constexpr int n_bpc = 1;
constexpr int n_cpg = 1;


__global__ void base_matmul(__grid_constant__ const CUtensorMap A_map, __grid_constant__ const CUtensorMap B_map, nv_bfloat16*C)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("TMA Map Address: %p, %p\n", &A_map, &B_map);
    }
}

int main()
{
  NaiveTensor<nv_bfloat16> A(m*k); 
  NaiveTensor<nv_bfloat16> B(k*n);
  NaiveTensor<nv_bfloat16> C(m*n);

  A.allocate();
  B.allocate();
  C.allocate();
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  A.to_device();
  B.to_device();
  C.to_device();
  NaiveLauncher launcher(n_cpg, n_bpc,n_tpb);
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{m,k},{m,k});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(B.d_ptr, {k,n}, {k,n});
  launcher.launch(base_matmul, a_map, b_map, c.d_ptr);
  cudaDeviceSynchronize();



}
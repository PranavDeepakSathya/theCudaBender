#include "atoms/utils.cuh"
#include "atoms/naive_launcher.cuh"
#include "atoms/tma_factory.cuh"
#include "atoms/naive_tensor.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

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
  __shared__ alignas(128) nv_bfloat16 As[m][k]; 
  __shared__ alignas(128) nv_bfloat16 Bs[k][n];

  __shared__ barrier bar; 
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  barrier::arrival_token token; 
  if (is_elected())
  {
    int32_t coords[2] = {0,0};
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &As, &A_map, coords, cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &Bs, &B_map, coords, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As)+sizeof(Bs));
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token));
  
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
  launcher.launch(base_matmul, a_map, b_map, C.d_ptr);
  cudaDeviceSynchronize();



}
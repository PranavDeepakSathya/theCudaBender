#include "../atoms/utils.cuh"
#include "../atoms/naive_launcher.cuh"
#include "../atoms/tma_factory.cuh"
#include "../atoms/naive_tensor.cuh"

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


__global__ void base_matmul(__grid_constant__ const CUtensorMap A_map, __grid_constant__ const CUtensorMap B_map,
  NaiveTensor<float>::DeviceView C)
{
  __shared__ alignas(128) nv_bfloat16 As[m*k]; 
  __shared__ alignas(128) nv_bfloat16 Bs[k*n];

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
  if (threadIdx.x < 128)
  {
    //warning, if you want threads to acess consecutive threads to consecutive elements in row major 
    //the smem layout must be row major, but the thread layout must also be row major 
    //which is why at0 and at1 are calculated to be the standard lexical situation, 
    // if you want threads to access consecutive threeads to consecutive elements in col major 
    //the smem layout must be col major and the thread layout must also be col major. 
    
    int t = threadIdx.x; 
    int at0 = (t/k);
    int at1 = t%k;
    int bt0 = (t%k); 
    int bt1 = ((int)t/k) % n; 
    int a_off = k*at0 + at1; 
    int b_off = bt0 + n*at1; 
    nv_bfloat16 va = As[a_off];
    nv_bfloat16 vb = Bs[b_off];

    printf("As (row_major): t: %d, -> (%d,%d) --> %d, va: %f ||| \n",t,at0,at1,a_off, (float) va);
    printf("Bs (col_major): t: %d, -> (%d,%d) --> %d, vb: %f ||| \n",t,bt0,bt1,b_off, (float) vb);
  }

}

int main()
{
  NaiveTensor<nv_bfloat16> A({m,k}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16> B({k,n}, Layout::COL_MAJOR);
  NaiveTensor<float> C({m,n}, Layout::ROW_MAJOR);

  A.allocate();
  B.allocate();
  C.allocate();
  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  A.to_device();
  B.to_device();
  C.to_device();
  auto C_view = C.get_device_view();
  NaiveLauncher launcher(n_cpg, n_bpc,n_tpb);
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{m,k},{m,k});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {k,n}, {k,n});
  launcher.launch(base_matmul, a_map, b_map, C_view);
  cudaDeviceSynchronize();



}
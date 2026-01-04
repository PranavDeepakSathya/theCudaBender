#include "atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;
namespace atmos = ld_st_mat_atom;
constexpr int wm = 16; 
constexpr int wn = 16; 
constexpr int wk = 16; 
constexpr int k_stages = 4; 
constexpr int warp_m = 4; 
constexpr int warp_n = 4; 
constexpr int m = warp_m * wm; 
constexpr int n = warp_n * wn; 
constexpr int k = k_stages * wk; 

__global__ void k_cycle(
    __grid_constant__ const CUtensorMap A_map,
    __grid_constant__ const CUtensorMap B_map,
    NaiveTensor<nv_bfloat16>::DeviceView A_out,
    NaiveTensor<nv_bfloat16>::DeviceView B_out
)
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

  
}

int main()
{
    NaiveTensor<nv_bfloat16> A({m,k}, Layout::ROW_MAJOR); 
    NaiveTensor<nv_bfloat16> B({k,n}, Layout::COL_MAJOR);
    NaiveTensor<nv_bfloat16> A_out({m,k}, Layout::ROW_MAJOR); 
    NaiveTensor<nv_bfloat16> B_out({k,n}, Layout::COL_MAJOR);

    A.allocate(); B.allocate(); A_out.allocate(); B_out.allocate();
    
    A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
    B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    
    A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
    B_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

    A.to_device(); 
    B.to_device();
    A_out.to_device(); 
    B_out.to_device();

    auto A_out_view = A_out.get_device_view();
    auto B_out_view = B_out.get_device_view();
    
    CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{m,k},{m,k});
    CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {k,n}, {k,n});

    dim3 threads(warp_m * warp_n * 32, 1, 1);
    dim3 blocks(1, 1, 1);
    
    int smem_size = 227 * 1024;
    cudaFuncSetAttribute(k_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Launching Kernel [M=" << m << ", N=" << n << ", K=" << k << "]..." << std::endl;

    k_cycle<<<blocks, threads, smem_size>>>(a_map, b_map, A_out_view, B_out_view);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    constexpr int ITERS = 20;
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < ITERS; ++i) {
        k_cycle<<<blocks, threads, smem_size>>>(a_map, b_map, A_out_view, B_out_view);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Avg Latency: " << ms / ITERS << " ms" << std::endl;

    std::cout << "Verifying..." << std::endl;
    A_out.to_host();
    B_out.to_host();

    int err_a = 0;
    int err_b = 0;
    
    for(size_t i = 0; i < A.size; ++i) {
        float val = static_cast<float>(A.h_ptr[i]);
        float out = static_cast<float>(A_out.h_ptr[i]);
        if (abs(val - out) > 1e-4) err_a++;
    }

    for(size_t i = 0; i < B.size; ++i) {
        float val = static_cast<float>(B.h_ptr[i]);
        float out = static_cast<float>(B_out.h_ptr[i]);
        if (abs(val - out) > 1e-4) err_b++;
    }

    if (err_a == 0 && err_b == 0) {
        std::cout << "SUCCESS: A and B match perfectly." << std::endl;
    } else {
        std::cout << "FAILURE: A_err=" << err_a << ", B_err=" << err_b << std::endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
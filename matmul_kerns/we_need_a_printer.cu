#include "../atoms/all.cuh"
constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 

constexpr int BK_stages = 2; //there thins ARE FIXED, cant change em ever 
//cause we are declaring static named regs. 
constexpr int acc_per_warp_m = 2; 
constexpr int acc_per_warp_n = 4; 

constexpr int BM = mma_m*acc_per_warp_m; 
constexpr int BN = mma_n*acc_per_warp_n; 
constexpr int BK_stage_iters = 16; 
constexpr int BK = mma_k*BK_stages*BK_stage_iters; 

constexpr size_t As_bytes = BM * BK * sizeof(nv_bfloat16);
constexpr size_t Bs_bytes = BK * BN * sizeof(nv_bfloat16); 
constexpr size_t shared_allocate_bytes = As_bytes + Bs_bytes + (4*128);


__global__ void warp_matmul (__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  NaiveTensor<float>::DeviceView C)

{

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  nv_bfloat16* As = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);
  smem = align128(smem);
  nv_bfloat16* Bs = (nv_bfloat16*)smem;

  uint32_t smem_base_a = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_b = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));

    __shared__ barrier bar; 

  if (l == 0)
  {
    init(&bar, threads_per_block); 
  }

  __syncthreads();

  barrier::arrival_token token; 
  
  if (is_elected())
  {
     

    int32_t coords_A[2] = {0,0};
    int32_t coords_B[2] = {0,0}; 

    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, As_bytes + Bs_bytes);
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); 

  
  


}

int main()
{
  return 0;
}

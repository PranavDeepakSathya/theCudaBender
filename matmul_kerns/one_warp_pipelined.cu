constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 

constexpr int BK_stages = 2; 
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

  

}

int main()
{

}

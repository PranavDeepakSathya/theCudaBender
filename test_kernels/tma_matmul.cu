#include "../atoms/all.cuh"
#include "sm_120_matmul_config.cuh"
using default_cfg = Sm120_BF16_Gemm_Config;
namespace ptx = cuda::ptx; 

template <class cfg>
__global__ void matmul
          (
            __grid_constant__ const CUtensorMap a_map, 
            __grid_constant__ const CUtensorMap b_map,
            float* C
          )
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As[cfg::bk_stages];
  nv_bfloat16* Bs[cfg::bk_stages]; 
  uint8_t* ptr = smem_raw;

  #pragma unroll 
  for (int stage = 0; stage < cfg::bk_stages; stage++)
  {
    As[stage] = alloc<nv_bfloat16,1024>(ptr, cfg::BM*cfg::BK); 
  }

  #pragma unroll 
  for (int stage = 0; stage < cfg::bk_stages; stage++)
  {
    Bs[stage] = alloc<nv_bfloat16,1024>(ptr, cfg::BK*cfg::BN); 
  }

  uint64_t* empty = alloc<uint64_t, 8>(ptr, cfg::bk_stages);
  uint64_t* full = alloc<uint64_t, 8>(ptr, cfg::bk_stages);
  uint64_t empty_tokens[cfg::bk_stages];
  uint64_t full_tokens[cfg::bk_stages];
  
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int b = blockIdx.x; 

  if (t == 0)
  {
    for (int stage = 0; stage < cfg::bk_stages; stage++)
    {
      ptx::mbarrier_init(&empty[stage],cfg::block_size);
      ptx::mbarrier_init(&full[stage],cfg::block_size);
  
    }
  }
  __syncthreads();
  
}

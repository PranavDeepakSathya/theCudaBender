#include "../atoms/all.cuh"

template <class Cfg> 
__global__ void matmul (__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB, float*gC)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];

  nv_bfloat16* As[Cfg::k_stages]; 
  nv_bfloat16* Bs[Cfg::k_stages]; 

  uint32_t smem_base_a[Cfg::k_stages];
  uint32_t smem_base_b[Cfg::k_stages];

  __shared__ barrier empty[Cfg::k_stages], full[Cfg::k_stages]; 

  int b = blockIdx.x; 
  int t = threadIdx.x; 
  int w = t/32; 
  int l = t % 32; 

  init_tiles_and_barriers<Cfg>(
    smem_raw,
    As, Bs,
    smem_base_a, smem_base_b,
    empty, full);

}



int main()
{
  return 0; 
}
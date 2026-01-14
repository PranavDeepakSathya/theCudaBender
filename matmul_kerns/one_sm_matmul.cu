#include "../atoms/all.cuh"

constexpr int block_M = 256; 
constexpr int block_N = 256; 
constexpr int block_K = 256; 
constexpr int K = 4096;
constexpr int n_producer_warps = 16; 
constexpr int n_comsumer_warps = 1; 
constexpr int n_tpb = 32*(n_comsumer_warps+n_producer_warps); 

__global__ void base_matmul(__grid_constant__ const CUtensorMap A_map, 
  __grid_constant__ const CUtensorMap B_map, 
  __grid_constant__ const CUtensorMap C_map)

{
  __shared__ alignas(128) nv_bfloat16 As0[block_M*block_K]; 
  __shared__ alignas(128) nv_bfloat16 Bs0[block_K*block_N];
  __shared__ alignas(128) nv_bfloat16 As1[block_M*block_K]; 
  __shared__ alignas(128) nv_bfloat16 Bs1[block_K*block_N];
  __shared__ alignas(128) float Cs[block_M*block_N];

  int t = threadIdx.x;
  int w = t/32;
  int l = t%32;

  __shared__ barrier bar; 
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();


}

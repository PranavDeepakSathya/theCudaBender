#include "../atoms/all.cuh"
#include "sm_120_matmul_config.cuh"

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


int main() 
{
  using Cfg = Sm120_BF16_Gemm_Config;

  NaiveTensor<nv_bfloat16> A({Cfg::M, Cfg::K}, Layout::ROW_MAJOR);
  NaiveTensor<nv_bfloat16> B({Cfg::K, Cfg::N}, Layout::COL_MAJOR);
  NaiveTensor<float>       C({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);

  A.allocate(); B.allocate(); C.allocate();

  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

  A.to_device();
  B.to_device();
  C.to_device();

  CUtensorMap a_map =
      TmaDescriptor<nv_bfloat16>::create_2d_row_major(
          A.d_ptr, {Cfg::M, Cfg::K}, {Cfg::BM, Cfg::BK});

  CUtensorMap b_map =
      TmaDescriptor<nv_bfloat16>::create_2d_col_major(
          B.d_ptr, {Cfg::K, Cfg::N}, {Cfg::BK, Cfg::BN});

  NaiveLauncher launcher(Cfg::grid_size, 1, Cfg::block_size, Cfg::shared_bytes);

  launcher.launch(matmul<Cfg>, a_map, b_map, C.d_ptr);

  cudaDeviceSynchronize();
  C.to_host();

  printf("Launch complete.\n");
}


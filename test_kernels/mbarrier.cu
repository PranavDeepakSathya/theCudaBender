#include "../atoms/all.cuh"
namespace ptx = cuda::ptx;
using barrier = cuda::barrier<cuda::thread_scope_block>;



using Cfg = GemmConfig<GemmInputs>;

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

  gC[t] = 1.0;

}


int main()
{
  return 0;
  NaiveTensor<nv_bfloat16>A({Cfg::problem_m,Cfg::problem_k}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>B({Cfg::problem_k,Cfg::problem_n}, Layout::COL_MAJOR); 
  NaiveTensor<float>C({Cfg::problem_m,Cfg::problem_n}, Layout::ROW_MAJOR); 
  A.allocate();
  B.allocate();
  C.allocate(); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); 
  B.to_device();
  C.to_device(); 
  CUtensorMap a_map = 
                    TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,
                    {Cfg::problem_m, Cfg::problem_k},{Cfg::BM,Cfg::BK});

  CUtensorMap b_map = 
                      TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, 
                        {Cfg::problem_k,Cfg::problem_n}, {Cfg::BK,Cfg::BN});

  NaiveLauncher launcher(Cfg::GM*Cfg::GN, 1, Cfg::block_threads,Cfg::smem_bytes);
  launcher.launch(matmul<Cfg>, a_map, b_map, C.d_ptr);
  cudaDeviceSynchronize();
  C.to_host();


  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      printf("%f, ", C.h_ptr[i*Cfg::problem_n + j]);
    }
    printf("\n");
  }


}
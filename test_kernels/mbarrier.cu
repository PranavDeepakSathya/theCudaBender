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

  init_tiles_and_barriers<Cfg>(
    smem_raw,
    As, Bs,
    smem_base_a, smem_base_b,
    empty, full);

  if (w == Cfg::n_consumer_warps)
  {
    for (int k_iter = 0; k_iter < Cfg::num_k_iters; k_iter++)
    {
      int stage = k_iter % 2; 
      empty[stage].wait(empty[stage].arrive());
      if (l == 0)
      {
        TmaLoadA<Cfg>::run(gA, As[stage], full[stage],b,k_iter);
        TmaLoadB<Cfg>::run(gB, Bs[stage], full[stage],b,k_iter);
        barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
        full[stage], 1, Cfg::As_bytes + Cfg::Bs_bytes);
      }
      else
      {
        barrier::arrival_token _ = full[stage].arrive();
      }
    }
  }

  else
  {
    #pragma unroll
    for (int i = 0; i < Cfg::k_stages; ++i) 
    {
          // i initially, all buffers are considered empty; ready for write
          barrier::arrival_token _ = empty[i].arrive();
    }
    uint32_t ra[Cfg::warp_m_tiles*4]; //the access is (wm,reg_id) -> 4*wm + reg_id 
    uint32_t rb[Cfg::warp_n_tiles*2]; //the access is (wn, reg_id) -> 2*wn + reg_id 
    float rc[Cfg::warp_m_tiles*Cfg::warp_n_tiles*4] = {0.0f}; //the access is (wm,wn,reg_id) -> acc_per_warp_n*wm*4 + wn*4 + reg_id 

    for (int k_iter = 0; k_iter < Cfg::num_k_iters; k_iter++)
    {
      int stage = k_iter % 2; 
      full[stage].wait(full[stage].arrive()); 
      for (int wk = 0; wk < Cfg::bk_mma_slices; wk++)
      {
        LdMatrixA<Cfg>::run(ra, smem_base_a[stage],wk,w,l);
        LdMatrixB<Cfg>::run(rb, smem_base_b[stage],wk,w,l);
        MmaLoop<Cfg>::run(rc,ra,rb);
      }

      barrier::arrival_token _ = empty[stage].arrive();
    }

    CStore<Cfg>::run(gC, b,w,l,rc);
  }

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
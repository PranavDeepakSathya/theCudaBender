#include "../atoms/all.cuh"
constexpr const int M = 512; 
constexpr const int N = 512; 
constexpr const int BM = 32; 
constexpr const int BN = 32; //32*32 = 1024 elements = 2048 bytes, make sure we dont fuck aligment up 
//by making this smaller than 1024 bytes. 

constexpr const int num_stages = 4;
constexpr const uint32_t shared_bytes = (num_stages*BM*BN*sizeof(nv_bfloat16)) + 4*1024; 
constexpr const uint32_t tile_size = BM*BN*sizeof(nv_bfloat16);
constexpr const int BM_iters = M/BM; 
constexpr const int BN_iters = N/BN; 
constexpr const int total_iters = BM_iters*BN_iters; 


__global__ void copy_kern(__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap a_out_map)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As[num_stages]; 
  uint8_t* ptr = smem_raw;
  int t = threadIdx.x; 
  int w = t/32;
  int l = t % 32; 

  for (int s = 0; s < num_stages; s++)
  {
    As[s] = alloc<nv_bfloat16, 128>(ptr, BM*BN);
  }
  uint64_t* empty = alloc<uint64_t, 16>(ptr, num_stages);
  uint64_t* full = alloc<uint64_t, 16>(ptr, num_stages);

  if (t == 0)
  {
    for (int s = 0; s < num_stages; s++)
    {
      cuda::ptx::mbarrier_init(empty[s],2);
      cuda::ptx::mbarrier_init(full[s],2);

    }
  }

  if (w == 0) //producer code 
  {
    for (int idx = 0; idx < total_iters; idx ++)
    {
      int stage = idx % num_stages; 

      uint32_t coords = {(idx / BN_iters)*BM, (idx % BN_iters)*BN}
      
    }
  }
}

int main()
{
  NaiveTensor<nv_bfloat16>A({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>A_out({M,N},Layout::ROW_MAJOR); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device();
  A_out.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{M,N},{BM,BN});
  CUtensorMap a_out_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A_out.d_ptr,{M,N},{BM,BN});

  NaiveLauncher launcher(1, 1, 64,shared_bytes);
  launcher.launch(copy_kern, a_map, a_out_map);
  cudaDeviceSynchronize(); 
  A_out.to_host(); 
  
  printf("\n ====================A================\n");
  A.pretty_print(); 
  printf("\n ====================A_out ================\n");
  A_out.pretty_print(); 


 
}
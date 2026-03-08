#include "../atoms/all.cuh"
constexpr int m = 16; 
constexpr int n = 16; 
constexpr int grid_size = 1; 
constexpr int block_size = 32; 
namespace wa = warp_function;


__global__ void transpose_kernel(__grid_constant__ const CUtensorMap a_map,
                                __grid_constant__ const CUtensorMap a_out_map)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t As = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); 
  uint32_t As_out = As + (m*n*sizeof(nv_bfloat16)); 
  uint32_t bar = As_out + (m*n*sizeof(nv_bfloat16)); 
  uint32_t ra[4];

  int l = threadIdx.x; 
  if (l==0) mbarrier_init(bar,32); 
  __syncthreads(); 

  if (l==0)
  {
    cp_async_bulk_tensor_2d(As, &a_map, 0,0, bar);
    mbarrier_arrive_expect_tx(bar,m*n*sizeof(nv_bfloat16)); 
  }
  else mbarrier_arrive(bar); 


  mbarrier_wait_parity(bar,0); 
  uint32_t a_ld_addr = As + (((l%16)*n + 8*(l/16))*sizeof(nv_bfloat16)); 
  uint32_t a_st_addr = As_out + (((l%16)*m + 8*(l/16))*sizeof(nv_bfloat16)); 
  wa::ldmatrix_m8n8_x4_trans_b16(ra, a_ld_addr); 
  // after ldmatrix
  uint32_t r0 = ra[0];
  uint32_t r1 = ra[1];

  // unpack bf16 (2 per 32-bit reg)
  uint16_t lo0 = r0 & 0xFFFF;
  uint16_t hi0 = r0 >> 16;
  uint16_t lo1 = r1 & 0xFFFF;
  uint16_t hi1 = r1 >> 16;

  // convert to float
  float f0 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(&lo0));
  float f1 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(&hi0));
  float f2 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(&lo1));
  float f3 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(&hi1));

  // ordered print
  for (int i = 0; i < 32; i++) {
    __syncthreads();
    if (threadIdx.x == i) {
      printf("lane %d: %f %f %f %f\n", i, f0, f1, f2, f3);
    }
  }
  __syncthreads();

  wa::stmatrix_m8n8_x4_b16(ra, a_st_addr);
  tma_fence();

  cp_async_bulk_tensor_2d_store(&a_out_map,0,0,As_out);
  cp_async_commit_group();
  cp_async_wait_group<0>(); 

}

int main()
{
  NaiveTensor<nv_bfloat16>A({m,n},Layout::ROW_MAJOR);    

  NaiveTensor<nv_bfloat16>A_out({m,n},Layout::ROW_MAJOR); 

  A.allocate(); A_out.allocate(); 
  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); A_out.to_device(); 
  A.pretty_print();
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{m,n},{m,n});
  CUtensorMap a_out_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A_out.d_ptr,{m,n},{m,n});

  NaiveLauncher launcher(grid_size,1,block_size, (2*m*n*sizeof(nv_bfloat16)) + 1024); 
  launcher.launch(transpose_kernel, a_map, a_out_map);
  cudaDeviceSynchronize(); 
  A_out.to_host(); 

  A_out.pretty_print();
}


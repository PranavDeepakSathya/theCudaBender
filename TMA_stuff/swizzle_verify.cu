#include "../atoms/all.cuh"
constexpr int M = 128;
constexpr int N = 64; 
constexpr int atom_m = 16;
constexpr int atom_n = 16; 
constexpr int m_iters = M/atom_m; 
constexpr int n_iters = N/atom_n;
constexpr int b_bits = 3;
constexpr int m_base = 4; 
constexpr int s_shift = 3;
constexpr CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_128B;
namespace ptx = cuda::ptx; 
using barrier = cuda::barrier<cuda::thread_scope_block>;


__global__ void verify_swizzle(const __grid_constant__ CUtensorMap a_map,
                const __grid_constant__ CUtensorMap a_out_map)
{
  
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* ptr = smem_raw;
  nv_bfloat16* As = alloc<nv_bfloat16, 1024>(ptr, M*N);
  nv_bfloat16* As_out = alloc<nv_bfloat16,1024>(ptr, M*N);
  uint32_t as_base = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t as_out_base = static_cast<uint32_t>(__cvta_generic_to_shared(As_out));

    int l = threadIdx.x; 
    int l_row = l%16;
    int l_col = 8*(l/16);
    uint32_t ra[4];

  __shared__ barrier bar; 
  barrier::arrival_token token;

  if (l == 0)
    {
      init(&bar, blockDim.x);
    }
    __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
  int32_t A_coords[2] = {0, 0};


  if(l == 0)
  {
    ptx::cp_async_bulk_tensor(
    ptx::space_shared, ptx::space_global,
      As, &a_map, A_coords, cuda::device::barrier_native_handle(bar));

    token = cuda::device::barrier_arrive_tx(bar, 1, M*N*sizeof(nv_bfloat16));
    
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); 
  asm volatile("fence.proxy.async.shared::cta;");

  for (int m_idx = 0; m_idx < M; m_idx += atom_m)
  {
    for (int n_idx = 0; n_idx < N; n_idx += atom_n)
    {
      int offset = (m_idx + l_row)*N + (n_idx + l_col); 
      uint32_t swizzle_offset_bytes = cute_swizzle_byte_offset<b_bits,m_base,s_shift,nv_bfloat16>(offset);
      warp_atom::ldmatrix_m8n8_x4_b16(ra[0],ra[1],ra[2],ra[3], as_base + swizzle_offset_bytes);

      __syncthreads();
      warp_atom::stmatrix_m8n8_x4_b16(ra[0],ra[1],ra[2],ra[3], as_out_base + swizzle_offset_bytes);

    }
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");


  if(l == 0)
  {
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_global, cuda::ptx::space_shared,
        &a_out_map, A_coords, As_out
      );
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  }
}

int main()
{
  NaiveTensor<nv_bfloat16>A({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>A_out({M,N}, Layout::ROW_MAJOR); 
  A.allocate();
  A_out.allocate();


  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS,DIST_FLOAT_NEG1_1); 
  A.to_device();
  A_out.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr, {M,N},{M,N}, swizzle_mode);
  CUtensorMap a_out_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A_out.d_ptr, {M,N},{M,N}, swizzle_mode);
  NaiveLauncher launcher(1,1,32,2*M*N*sizeof(nv_bfloat16) + 4*1024);
  launcher.launch(verify_swizzle, a_map, a_out_map);
  cudaDeviceSynchronize();
  A_out.to_host(); 

  double max_abs_err = 0.0;
  int bad = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {

      float a  = __bfloat162float(A.h_ptr[i*N + j]);
      float ao = __bfloat162float(A_out.h_ptr[i*N + j]);

      double err = fabs(a - ao);
      max_abs_err = std::max(max_abs_err, err);

      if (err > 1e-2) {
        bad++;
        if (bad < 10) {
          printf("Mismatch at (%d,%d): A=%f  Out=%f  err=%f\n",
                i, j, a, ao, err);
        }
      }
    }
  }

  printf("\n==== Swizzle Verification ====\n");
  printf("Max abs error : %e\n", max_abs_err);
  printf("Bad elements  : %d / %d\n", bad, M*N);

  if (bad == 0)
    printf("✅ PASSED\n");
  else
    printf("❌ FAILED\n");

}
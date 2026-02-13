#include "../atoms/all.cuh"
constexpr int M = 128;
constexpr int N = 128; 
constexpr int out_m = 16;
constexpr int out_n = 16; 
constexpr int m_iters = M/out_m; 
constexpr int n_iters = N/out_n; 
constexpr int bench_iters = 20000000; 
constexpr int TILE_M = out_m; // 16
constexpr int TILE_N = out_n; // 16
constexpr int b_bits = 3;
constexpr int m_base = 4;
constexpr int s_shift = 3;
constexpr int NUM_TILES_M = m_iters; // 8
constexpr int NUM_TILES_N = n_iters; // 8
constexpr int NUM_TILES   = NUM_TILES_M * NUM_TILES_N;
constexpr int TN_SHIFT = 3;                 // log2(NUM_TILES_N)
constexpr int TN_MASK  = NUM_TILES_N - 1; 

namespace ptx = cuda::ptx; 
using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __forceinline__
int next_tile_id(int tile_id, const uint32_t ra[4])
{
    // Mix the fragment regs into some entropy
    uint32_t mix =
        ra[0] ^
        (ra[1] * 0x9e3779b9u) ^
        (ra[2] * 0x85ebca6bu) ^
        (ra[3] * 0xc2b2ae35u);

    // Step in [0,63]
    tile_id = (tile_id + (mix & (NUM_TILES - 1))) & (NUM_TILES - 1);
    return tile_id;
}

__global__ void test_ldmx4(const __grid_constant__ CUtensorMap a_map, 
  nv_bfloat16 *out, unsigned long long *clock, unsigned long long *clock_swiz)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* ptr = smem_raw;
 
  nv_bfloat16* As = alloc<nv_bfloat16,1024>(ptr, M*N);
  int t = threadIdx.x; 
  uint32_t ra[4]; 

  int lane_row = t%16;
  int lane_col = 8*(t/16);
 
  uint32_t smem_addr =
  static_cast<uint32_t>(__cvta_generic_to_shared(As));

  __shared__ barrier bar; 
  barrier::arrival_token token;
  if (t == 0)
    {
      init(&bar, blockDim.x);
    }
    __syncthreads();

    if (t == 0)
    {
      int32_t A_coords[2] = {0,0};
      

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

  int tile_id0 = 0;
  uint32_t sink0 = 0;

  unsigned long long start0 = clock64();

  #pragma unroll 
  for (int iter = 0; iter < bench_iters; iter++)
  {
    int offset = ((tile_id0/n_iters)*TILE_M + lane_row)*N + ((tile_id0%n_iters)*TILE_N + lane_col); 


    warp_atom::ldmatrix_m8n8_x4_b16(
        ra[0], ra[1], ra[2], ra[3],
        smem_addr + (offset*sizeof(nv_bfloat16)));

  
    tile_id0 = next_tile_id(tile_id0, ra);
    sink0 ^= ra[0];
  }

  
  unsigned long long end0 = clock64();

  int tile_id1 = 0;
  uint32_t sink1 = 0;

  unsigned long long start1 = clock64();

  #pragma unroll 
  for (int iter = 0; iter < bench_iters; iter++)
  {
    
    int offset = ((tile_id1/n_iters)*TILE_M + lane_row)*N + ((tile_id1%n_iters)*TILE_N + lane_col); 
    uint32_t swizzled_byte_offset = cute_swizzle_byte_offset<b_bits,m_base,s_shift,nv_bfloat16>(offset);
    warp_atom::ldmatrix_m8n8_x4_b16(
        ra[0], ra[1], ra[2], ra[3],
        smem_addr + swizzled_byte_offset);

    

    tile_id1 = next_tile_id(tile_id1, ra);
    sink1 ^= ra[0];
  }
  

  unsigned long long end1 = clock64();


  if (t == 0) {
    clock[0]      = end0 - start0;
    clock_swiz[0] = end1 - start1;
    out[0] = __float2bfloat16(float(sink0 ^ sink1));
  }
    
}

int main()
{
  NaiveTensor<nv_bfloat16>A({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>A_out({out_m,out_n}, Layout::ROW_MAJOR); 
  NaiveTensor<unsigned long long>clock({1}, Layout::ROW_MAJOR);
  NaiveTensor<unsigned long long>clock_swiz({1},Layout::ROW_MAJOR);
  A.allocate();
  A_out.allocate();
  clock.allocate();
  clock_swiz.allocate();
  A.to_device(); 
  A_out.to_device(); 
  clock.to_device();
  clock_swiz.to_device();

  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr, {M,N}, {M,N});
  NaiveLauncher launcher(1,1,32,2*M*N*sizeof(nv_bfloat16) + 4*1024);

  launcher.launch(test_ldmx4, a_map, A_out.d_ptr, clock.d_ptr,clock_swiz.d_ptr);
  cudaDeviceSynchronize();
  A_out.to_host();
  clock.to_host(); 
  clock_swiz.to_host();
    printf("No-swizzle cycles:  %llu\n", clock.h_ptr[0]);
  printf("Swizzled cycles:    %llu\n", clock_swiz.h_ptr[0]);

  double cyc0 = double(clock.h_ptr[0]) / bench_iters;
  double cyc1 = double(clock_swiz.h_ptr[0]) / bench_iters;

  printf("Cycles/ldmatrix no-swiz = %.4f\n", cyc0);
  printf("Cycles/ldmatrix swiz    = %.4f\n", cyc1);
}
#include "atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int n_warps = 32; 
constexpr int warp_m = 8; 
constexpr int warp_n = n_warps/warp_m; // 4
constexpr int ld_m = 8; 
constexpr int ld_n = 8; 
constexpr int m = warp_m*ld_m; // 64
constexpr int n = warp_n*ld_n; // 32
constexpr int n_tpb = n_warps*32;


__global__ void test_ldm_row_maj_x1 (__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap a_out_map)
{
  __shared__ alignas(128) nv_bfloat16 As[m*n]; 
  __shared__ alignas(128) nv_bfloat16 A_os[m*n];

  __shared__ barrier bar; 
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  barrier::arrival_token token; 
  if (is_elected())
  {
    int32_t coords[2] = {0,0};
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &As, &a_map, coords, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As));
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  // --- Address Logic ---
  int t = threadIdx.x; 
  int w = t/32; 
  int lane = t % 32; 

  int w_m = w/warp_n; 
  int w_n = w % warp_n;
  
  // Base offsets for this warp's 8x8 tile
  int w_m_offset = w_m * ld_m; 
  int w_n_offset = w_n * ld_n; 
  
  // Stride (Row Major)
  int stride = n; 

  // ldmatrix specific: Lane K (0-7) must point to Row K of the tile.
  // We calculate the pointer for the specific row this lane "controls".
  // (Lanes 8-31 calculate valid addresses too, but hardware ignores them)
  int my_row_idx = w_m_offset + (lane % 8); 
  int my_col_idx = w_n_offset;
  
  int flat_offset = (my_row_idx * stride) + my_col_idx;

  // Convert to SMEM pointers
  uint32_t smem_base_A = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_Out = static_cast<uint32_t>(__cvta_generic_to_shared(A_os));
  
  uint32_t ld_addr = smem_base_A + (flat_offset * sizeof(nv_bfloat16));
  uint32_t st_addr = smem_base_Out + (flat_offset * sizeof(nv_bfloat16));

  // --- Registers ---
  nv_bfloat162 rA[1]; // Holds 2x bf16 (32 bits total)

  // --- LDMATRIX (Load 8x8 tile to regs) ---
  // Cast register to uint32_t for inline asm
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rA);
  
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
    : "=r"(reg_ptr[0]) 
    : "r"(ld_addr)
  );

  // --- STMATRIX (Store regs to 8x8 tile) ---
  // Mirrors ldmatrix. Lane K provides address for Row K write.
  asm volatile(
    "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
    : 
    : "r"(st_addr), "r"(reg_ptr[0])
    : "memory"
  );

  // --- Writeback to GMEM (TMA) ---
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  
  if (is_elected()) {
        int32_t coords_a[2] = {0, 0}; 
        ptx::cp_async_bulk_tensor(
            ptx::space_global, ptx::space_shared, 
            &a_out_map, coords_a, &A_os
        );
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
    
    if (threadIdx.x == 0) {
      (&bar)->~barrier();
    }
}


int main()
{
  NaiveTensor <nv_bfloat16> A({m,n}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16> A_out({m,n}, Layout::ROW_MAJOR);
  A.allocate(); A_out.allocate(); 
  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); A_out.to_device();

  CUtensorMap A_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr, {m,n}, {m,n});
  CUtensorMap A_out_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A_out.d_ptr, {m,n}, {m,n});

  test_ldm_row_maj_x1<<<1, n_tpb>>>(A_map, A_out_map);
  cudaDeviceSynchronize(); 

  A.to_host(); A_out.to_host(); 
  printf("---------------A_before_kernel-------------------\n");
  A.pretty_print(); 
    printf("---------------A_after_kernel-------------------\n");
  A_out.pretty_print(); 
}
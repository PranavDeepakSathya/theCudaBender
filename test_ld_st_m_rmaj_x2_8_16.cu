#include "atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

// --- Configuration for 16 Warps & .x2 (16x8 tile) ---
constexpr int n_warps = 16; 

// Grid Layout: 4x4 warps
constexpr int warps_m = 4;      
constexpr int warps_n = 4;      

// Tile Size per Warp (x2 = 16 rows)
constexpr int ld_m = 8;        // 16 rows per warp
constexpr int ld_n = 16;         // 8 cols per warp
constexpr int lane_m = 8;
constexpr int lane_n = 8;
// Total Tensor Size
constexpr int m = warps_m * ld_m; // 4 * 16 = 64
constexpr int n = warps_n * ld_n; // 4 * 8  = 32

constexpr int n_tpb = n_warps * 32;

__global__ void test_ldm_row_maj_x2 (__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap a_out_map)
{
  __shared__ alignas(128) nv_bfloat16 As[m*n]; 
  __shared__ alignas(128) nv_bfloat16 A_os[m*n];

  __shared__ barrier bar; 
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // --- 1. Load Global -> Shared (TMA) ---
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

  // --- 2. Address Generation (.x2 Logic) ---
  int t = threadIdx.x; 
  int w = t / 32; 
  int lane = t % 32; 

  // Map Warp ID to Grid (4x4)
  int w_m = w / warps_n; 
  int w_n = w % warps_n;
  
  // Base offsets for this warp's 16x8 tile
  int w_m_base = w_m * ld_m; 
  int w_n_base = w_n * ld_n; 
  
  int stride = n; 

  // --- THE DEEP PART ---
  // ldmatrix.x2 reads 16 addresses from Lanes 0-15.
  // Lane 0 points to Row 0 of the tile.
  // Lane 15 points to Row 15 of the tile.
  // Lanes 16-31 are ignored for addressing.
  // to switch from 16x8 to 8x16, we know that the first 8 bits should be the first 8 rows, 
  // the next 8 bits should be the 
  int lane_row_offset = lane % 8;
  int lane_col_offset = lane_n*(lane / 8); 

  int my_row_idx = w_m_base + lane_row_offset; 
  int my_col_idx = w_n_base + lane_col_offset;
  
  
  int flat_offset = (my_row_idx * stride) + my_col_idx;

  // Convert to SMEM pointers
  uint32_t smem_base_A = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_Out = static_cast<uint32_t>(__cvta_generic_to_shared(A_os));
  
  uint32_t ld_addr = smem_base_A + (flat_offset * sizeof(nv_bfloat16));
  uint32_t st_addr = smem_base_Out + (flat_offset * sizeof(nv_bfloat16));

  // --- 3. Registers (.x2 needs 2x 32-bit registers) ---
  nv_bfloat162 rA[2]; 

  // --- 4. LDMATRIX.x2 (Load 16x8 tile) ---
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rA);
  
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
    : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]) 
    : "r"(ld_addr)
  );

  // --- 5. STMATRIX.x2 (Store 16x8 tile) ---
  asm volatile(
    "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
    : 
    : "r"(st_addr), "r"(reg_ptr[0]), "r"(reg_ptr[1])
    : "memory"
  );

  // --- 6. Writeback ---
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

  test_ldm_row_maj_x2<<<1, n_tpb>>>(A_map, A_out_map);
  cudaDeviceSynchronize(); 

  A.to_host(); A_out.to_host(); 
  printf("---------------A_before_kernel-------------------\n");
  A.pretty_print(); 
    printf("---------------A_after_kernel-------------------\n");
  A_out.pretty_print(); 
}
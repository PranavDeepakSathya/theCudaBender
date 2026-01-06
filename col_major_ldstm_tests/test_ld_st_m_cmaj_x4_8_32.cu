#include "atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

// --- Configuration for 8x32 Tile using .x4 ---
constexpr int n_warps = 8; 

// Grid Layout: 2x2 warps
constexpr int warps_m = 4;      
constexpr int warps_n = 2;      

// Tile Size per Warp
// Standard .x4 is 32x8. We are re-stitching it to 8x32.
constexpr int ld_m = 8;          // 8 rows
constexpr int ld_n = 32;         // 32 columns 
constexpr int lane_n = 8;        // Width of one ldmatrix segment

// Total Tensor Size
constexpr int m = warps_m * ld_m; // 2 * 8 = 16
constexpr int n = warps_n * ld_n; // 2 * 32 = 64

constexpr int n_tpb = n_warps * 32;

__global__ void test_ldm_row_maj_x4 (__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap a_out_map)
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

  // --- 2. Address Generation (.x4 Logic) ---
  int t = threadIdx.x; 
  int w = t / 32; 
  int lane = t % 32; 

  // Map Warp ID
  int w_m = w / warps_n; 
  int w_n = w % warps_n;
  
  // Base offsets for this warp's 8x32 tile
  int w_m_base = w_m * ld_m; 
  int w_n_base = w_n * ld_n; 
  
  int stride = m; 

  // --- THE DEEP PART: 8x32 Mapping ---
  // We have 32 Lanes (0-31).
  // We want 4 horizontal blocks (0-7, 8-15, 16-23, 24-31 columns).
  // Each block needs 8 addresses (Rows 0-7).
  
  // 1. Which block am I? (0, 1, 2, or 3)
  int block_idx = lane / 32; 
  
  // 2. Which row in that block? (0-7)
  int lane_col_offset = lane % 32;
  
  // 3. Calculate Column Offset based on block
  int lane_row_offset = block_idx * lane_n; // 0, 8, 16, or 24

  int my_row_idx = w_m_base; 
  int my_col_idx = w_n_base + lane_col_offset;
  
  int flat_offset = (my_row_idx) + (my_col_idx* stride);

  // Convert to SMEM pointers
  uint32_t smem_base_A = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_Out = static_cast<uint32_t>(__cvta_generic_to_shared(A_os));
  
  uint32_t ld_addr = smem_base_A + (flat_offset * sizeof(nv_bfloat16));
  uint32_t st_addr = smem_base_Out + (flat_offset * sizeof(nv_bfloat16));

  // --- 3. Registers (.x4 needs 4x 32-bit registers) ---
  nv_bfloat162 rA[4]; 

  // --- 4. LDMATRIX.x4 (Load 8x32 tile) ---
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rA);
  
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
    : "r"(ld_addr)
  );

  // --- 5. STMATRIX.x4 (Store 8x32 tile) ---
  asm volatile(
    "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
    : 
    : "r"(st_addr), "r"(reg_ptr[0]), "r"(reg_ptr[1]), "r"(reg_ptr[2]), "r"(reg_ptr[3])
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
  NaiveTensor <nv_bfloat16> A({m,n}, Layout::COL_MAJOR); 
  NaiveTensor<nv_bfloat16> A_out({m,n}, Layout::COL_MAJOR);
  A.allocate(); A_out.allocate(); 
  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); A_out.to_device();

  CUtensorMap A_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(A.d_ptr, {m,n}, {m,n});
  CUtensorMap A_out_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(A_out.d_ptr, {m,n}, {m,n});

  test_ldm_row_maj_x4<<<1, n_tpb>>>(A_map, A_out_map);
  cudaDeviceSynchronize(); 

  A.to_host(); A_out.to_host(); 
  printf("---------------A_before_kernel-------------------\n");
  A.pretty_print(); 
    printf("---------------A_after_kernel-------------------\n");
  A_out.pretty_print(); 
}
#include "../atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int mma_m = 16; 
constexpr int mma_k = 16; 
constexpr int mma_n = 8; 

constexpr int n_warps = 1; 
constexpr int threads_per_block = 32; 
constexpr int n_blocks = 1; 

 
constexpr int frags_n = 2;
constexpr int frags_m = 2; 


constexpr int n_iters_k = 8; 
constexpr int BM = frags_m*mma_m; 
constexpr int BN = frags_n*mma_n; 
constexpr int BK = n_iters_k*mma_k; 

constexpr size_t As_bytes = BM * BK * sizeof(nv_bfloat16);
constexpr size_t Bs_bytes = BK * BN * sizeof(nv_bfloat16); 
constexpr size_t shared_allocate_bytes = As_bytes + Bs_bytes + (4*128);

__global__ void k_cycled_matmul (__grid_constant__ const CUtensorMap gA, __grid_constant__ const CUtensorMap gB,
  NaiveTensor<float>::DeviceView C)

{
  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  nv_bfloat16* As = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);
  smem = align128(smem);
  nv_bfloat16* Bs = (nv_bfloat16*)smem;

  uint32_t smem_base_a = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_b = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));

  uint32_t ra[4][4]; 
  uint32_t rb[4][2];
  float rc[4][4];



  int l = threadIdx.x; 


  int a_lane_group_16x16_offset = 8*(l/16); 
  int a_lane_row_id = l%16; 

  int b_lane_group_16x8_offset = 8*(l/8); 
  int b_lane_col_id = l%8; 



  
  __shared__ barrier bar; 

  if (l == 0)
  {
    init(&bar, threads_per_block); 
  }

  __syncthreads();

  barrier::arrival_token token; 
  
  if (is_elected())
  {
     

    int32_t coords_A[2] = {0,0};
    int32_t coords_B[2] = {0,0}; 

    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, As_bytes + Bs_bytes);
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); 

  for (int warp_k_start = 0; warp_k_start < BK; warp_k_start += mma_k)
  {
    int comp_0_a_load_m = a_lane_row_id + 0;
    int comp_1_a_load_m = a_lane_row_id + 0;
    int comp_2_a_load_m = a_lane_row_id + mma_m; 
    int comp_3_a_load_m = a_lane_row_id + mma_m; 

    int comp_0_b_load_n = b_lane_col_id + 0; 
    int comp_1_b_load_n = b_lane_col_id + mma_n; 
    int comp_2_b_load_n = b_lane_col_id + 0; 
    int comp_2_b_load_n = b_lane_col_id + mma_n;

    int comp_0_a_load_k = a_lane_group_16x16_offset + warp_k_start;
    int comp_0_b_load_k = b_lane_group_16x8_offset + warp_k_start;


  }

  

}

int main()
{
  NaiveTensor<nv_bfloat16>A({BM,BK}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>B({BK,BN}, Layout::COL_MAJOR);
  NaiveTensor<float>C({BM,BN}, Layout::ROW_MAJOR); 
  A.allocate();
  B.allocate();
  C.allocate();
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device(); 
  B.to_device(); 
  C.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr, {BM,BK}, {BM,BK});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {BK,BN}, {BK,BN}); 
  auto c_view = C.get_device_view(); 

}


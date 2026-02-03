#include "../atoms/all.cuh"
constexpr int mma_m = 16; 
constexpr int mma_n = 8;
constexpr int mma_k = 16; 
constexpr int acc_per_warp_m = 2; 
constexpr int acc_per_warp_n = 2; 
constexpr int mma_k_iters = 4;
constexpr int WM = mma_m*acc_per_warp_m; 
constexpr int WN = mma_n*acc_per_warp_n;
constexpr int BK = mma_k*mma_k_iters;
constexpr int warps_per_block_m = 4; 
constexpr int warps_per_block_n = 6; 
constexpr int BM = warps_per_block_m*WM; 
constexpr int BN = warps_per_block_n*WN; 
constexpr int M = BM; 
constexpr int N = BN; 
constexpr int K = 4096; 

constexpr int n_producer_warps = 4; 
constexpr int n_consumer_warps = warps_per_block_m*warps_per_block_n;
constexpr int block_size = ((warps_per_block_m*warps_per_block_n) + (n_producer_warps))*32; 
constexpr int bk_stages = 3;

constexpr int prod_warp_start = (n_consumer_warps);
constexpr int p_thread_id = n_consumer_warps*32;
constexpr int num_BK_iters = K / BK;

constexpr uint32_t As_bytes = BM*BK*sizeof(nv_bfloat16);
constexpr uint32_t Bs_bytes = BK*BN*sizeof(nv_bfloat16); 
constexpr uint32_t shared_alloc_bytes = (bk_stages*(As_bytes + Bs_bytes)) + (bk_stages*2*128); 

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__global__ void matmul(__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  float* C)

{
  nv_bfloat16* As[bk_stages]; 
  nv_bfloat16* Bs[bk_stages]; 
  uint32_t smem_base_a[bk_stages];
  uint32_t smem_base_b[bk_stages];

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);
  allocate_smem_tiles(smem_raw, As_bytes, Bs_bytes, bk_stages, As, Bs, smem_base_a, smem_base_b); 
  __shared__ barrier full[bk_stages], empty[bk_stages];

  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int warp_start_m = (w / warps_per_block_n)*WM; 
  int warp_start_n = (w % warps_per_block_n)*WN; 

  if (threadIdx.x == 0) {
  for (int i = 0; i < bk_stages; ++i) {
      init(&full[i], (n_consumer_warps*32) + 1);
      init(&empty[i],(n_consumer_warps*32) + 1);
  }
  ptx::fence_proxy_async(ptx::space_shared);
}

  __syncthreads();

  //producer 
  if (w >= prod_warp_start)
  {
    if (t == p_thread_id)
    {
      int stage = 0; 
      for (int bk = 0; bk < num_BK_iters; ++bk, ++stage)
      {
        if (stage ==bk_stages) stage = 0;
        empty[stage].wait(empty[stage].arrive());

        int32_t coordsA[2] = {bk*BK, 0};
        int32_t coordsB[2] = {bk*BK, 0};

          ptx::cp_async_bulk_tensor(
              ptx::space_shared, ptx::space_global,
              As[stage], &gA, coordsA,
              cuda::device::barrier_native_handle(full[stage]));

          ptx::cp_async_bulk_tensor(
              ptx::space_shared, ptx::space_global,
              Bs[stage], &gB, coordsB,
              cuda::device::barrier_native_handle(full[stage]));

          barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
              full[stage], 1, As_bytes + Bs_bytes);

      }
    }
  }
  else//consumer logic 
  {
    for (int i = 0; i < bk_stages; ++i) {
          // i initially, all buffers are considered empty; ready for write
          barrier::arrival_token _ = empty[i].arrive();
      }
    
    int stage = 0; 

    int a_lane_row_base = l % 16;
    int a_lane_col_base = (l / 16) * 8;

    int b_lane_row_base = 8*(l/8);
    int b_lane_col_base = l % 8; 

    int32_t ra[acc_per_warp_m*4]; //the access is (wm,reg_id) -> 4*wm + reg_id 
    uint32_t rb[acc_per_warp_n*2]; //the access is (wn, reg_id) -> 2*wn + reg_id 
    float rc[acc_per_warp_m*acc_per_warp_n*4] = {0.0f}; //the access is (wm,wn,reg_id) -> acc_per_warp_n*wm*4 + wn*4 + reg_id 

    for (int bk = 0; bk < num_BK_iters; ++bk, ++stage)
    {
      if (stage == bk_stages) stage = 0;
      full[stage].wait(full[stage].arrive());
      
      #pragma unroll
      for (int wk = 0; wk < num_mma_k_iters; wk++)
      {
        #pragma unroll
        for (int wm = 0; wm < acc_per_warp_m; wm++)
        {
          int a_load_shared_row = warp_start_m + (wm*mma_m) + a_lane_row_base;
          int a_load_shared_col = (wk*mma_k) + a_lane_col_base;
          uint32_t a_ld_addr = (smem_base_a[stage]) + ((a_load_shared_col + (BK*a_load_shared_row))*sizeof(nv_bfloat16));
          int a_reg_start = wm*4; 
          warp_atom::ldmatrix_m8n8_x4_b16(ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3]);

          
        }

        #pragma unroll
        for (int wn = 0; wn < acc_per_warp_n; wn++)
        {
          int b_load_shared_row = (wk*mma_k) + b_lane_row_base; 
          int b_load_shared_col = warp_start_n + (wn*mma_n) + b_lane_col_base;
          uint32_t  b_ld_addr = (smem_base_b[stage]) + ((b_load_shared_row + (BK*b_load_shared_col))*sizeof(nv_bfloat16)); 
          int b_reg_start = wn*2; 
          warp_atom::ldmatrix_m8n8_x2_b16(rb[b_reg_start + 0], rb[b_reg_start + 1]);
          
        }
        
        #pragma unroll
        for (int wm = 0; wm < acc_per_warp_m; wm++)
        {
          #pragma unroll
          for (int wn = 0; wn < acc_per_warp_n; wn++)
          {
            int a_reg_start = wm*4; 
            int b_reg_start = wn*2; 
            int c_reg_start = (wm*acc_per_warp_n*4) + (wn*4); 
            warp_atom::void mma_m16n8k16_row_col_f32_bf16(
              rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3],
              ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3],
              rb[b_reg_start + 0], rb[b_reg_start + 1],
              rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3]
            );
          }
        }
        
      }
      
    }
  } 


}

int main()
{
  return 0;
}
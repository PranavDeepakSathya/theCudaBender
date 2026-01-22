#include "../atoms/all.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int M = 4096; 
constexpr int K = 4096; 
constexpr int N = 4096; 
constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 

constexpr int n_stages = 2; //this is fixed. 
constexpr int k_iters_per_stage = 4;

constexpr int consumer_warps_m = 4;
constexpr int consumer_warps_n = 4;

constexpr int n_consumer_warps =
    consumer_warps_m * consumer_warps_n;

constexpr int n_warps = n_consumer_warps + 1;
constexpr int block_size = n_warps * 32;

constexpr int BM = mma_m * consumer_warps_m;
constexpr int BN = mma_n * consumer_warps_n;
constexpr int BK = mma_k * k_iters_per_stage;


static_assert(M % BM == 0, "M must be divisible by BM (block M tile)");
static_assert(N % BN == 0, "N must be divisible by BN (block N tile)");
static_assert(K % (n_stages * BK) == 0,
    "K must be divisible by n_stages * BK");



__global__ void matmul (__grid_constant__ const CUtensorMap gA, __grid_constant__ const CUtensorMap gB,
  NaiveTensor<float>::DeviceView C)

{
  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  nv_bfloat16* As0 = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);

  smem = align128(smem);
  nv_bfloat16* As1 = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);

  smem = align128(smem);
  nv_bfloat16* Bs0 = (nv_bfloat16*)smem;
  smem += BK * BN * sizeof(nv_bfloat16);

  smem = align128(smem);
  nv_bfloat16* Bs1 = (nv_bfloat16*)smem;

  nv_bfloat16* As[2] = {As0, As1};
  nv_bfloat16* Bs[2] = {Bs0, Bs1};

  __shared__ barrier filled[n_stages];
  __shared__ barrier ready[n_stages];

  if (threadIdx.x < n_stages) 
  {
    init(&ready[threadIdx.x], blockDim.x);
    init(&filled[threadIdx.x], blockDim.x);
  }
  __syncthreads(); 

  // after init()

  __syncthreads();

  int num_steps = K / BK;
  int t = threadIdx.x;

  int w = t/32;
  int l = t%32;
  int b = blockIdx.x; 
  nv_bfloat162 ra[4]; 
  nv_bfloat162 rb[2]; 
  float rc[4];

    int gm = b / GN; 
  int gn = b % GN; 
  
  int block_m_start = gm*BM; 
  int block_n_start = gn*BN; 
  nv_bfloat162 ra[4]; 
  nv_bfloat162 rb[2]; 
  float rc[4] = {0.0};

  int wm = w / warps_n; 
  int wn = w % warps_n; 
  int warp_m_start = wm*mma_m; 
  int warp_n_start = wn*mma_n;

  int b_lane_group_16x8_id = l/8; 
  int b_lane_col_id = l % 8; 
  int b_lane_group_offset = b_lane_group_16x8_id*8; 
  int b_col_idx = warp_n_start + b_lane_col_id; 


  int a_lane_group_16x16_id = l/16; 
  int a_lane_row_id = l % 16;
  int a_lane_group_offset = a_lane_group_16x16_id*8; 
  int a_row_idx = warp_m_start + a_lane_row_id; 

  if (w == n_consumer_warps)
  {
    //producer: 
    for (int s = 0; s < num_steps, s++)
    {
    if (is_elected())
      {
        int block_k_start = s*BK; 

        int32_t coords_A[2] = {block_k_start, block_m_start};
        int32_t coords_B[2] = {block_k_start, block_n_start}; 

        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
        barrier_t::arrival_token token = cuda::device::barrier_arrive_tx(bar, 1, As_bytes + Bs_bytes);
      }
      else
      {
        barrier_t::arrival_token token = filled[i % 2].arrive();
      }

    }
  }


}

int main()
{
  return 0;
}
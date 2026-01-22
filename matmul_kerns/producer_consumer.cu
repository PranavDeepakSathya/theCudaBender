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


__device__ __forceinline__
void produce(
  nv_bfloat16** As,
  nv_bfloat16** Bs,
  int stage,
  int k,
  const CUtensorMap& gA,
  const CUtensorMap& gB,
  barrier* ready,
  barrier* empty)

{

}

__device__ __forceinline__
void consume(
    nv_bfloat16** As,
    nv_bfloat16** Bs,
    int stage,
    int warp_k_start,
    int lane_id,
    int warp_m_start,
    int warp_n_start,
    nv_bfloat162* ra,   // ← uses kernel registers
    nv_bfloat162* rb,
    float* rc
)
{
    // body later
}



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

  __shared__ barrier ready[n_stages];
  __shared__ barrier empty[n_stages];

  if (threadIdx.x < n_stages) 
  {
    init(&ready[threadIdx.x], blockDim.x);
    init(&empty[threadIdx.x], blockDim.x);
  }
  __syncthreads(); 

  // after init()
  empty[0].arrive();
  empty[1].arrive();
  __syncthreads();

  int num_steps = K / BK;
  int t = threadIdx.x;

  int w = t/32;
  int l = t%32;
  int b = blockIdx.x; 
  nv_bfloat162 ra[4]; 
  nv_bfloat162 rb[2]; 
  float rc[4];

  bool is_producer = w == n_consumer_warps; 

  if (is_producer)
  {
    empty[0].arrive_and_wait();

    produce(
      As,
      Bs,
      0,
      0,
      gA,
      gB,
      ready,
      empty
    );

    ready[0].arrive();
  }

  for (int s = 0; s < num_steps; s++)
  {

  }

}

int main()
{
  return 0;
}
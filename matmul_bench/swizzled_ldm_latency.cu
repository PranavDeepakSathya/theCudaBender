#include "../atoms/all.cuh"
namespace wa = warp_function; 
static constexpr int m = 256;
static constexpr int n = 64; 
static constexpr uint32_t ld_bytes = n*sizeof(nv_bfloat16);
static constexpr CUtensorMapSwizzle swizzle_mode =
      (ld_bytes == 32)  ? CU_TENSOR_MAP_SWIZZLE_32B  :
      (ld_bytes == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B  :
      (ld_bytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                          CU_TENSOR_MAP_SWIZZLE_NONE;

static_assert(swizzle_mode != CU_TENSOR_MAP_SWIZZLE_NONE);

static constexpr int swizzle_num =
    (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B)  ? 128  :
    (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B)  ? 384  :
    (swizzle_mode == CU_TENSOR_MAP_SWIZZLE_128B) ? 896 :
                                                   0;

static constexpr int iters = 20000;

static constexpr uint32_t smem_size = (m*n*sizeof(nv_bfloat16)) + 1024;
static constexpr uint32_t num_tiles = (m*n)/256;


__global__ void ldm_lat(__grid_constant__ const CUtensorMap a_map, uint64_t* out, uint32_t* sink)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint32_t As = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t bar = As + (m*n*sizeof(nv_bfloat16));
  uint32_t start_offset = 0;
  int l = threadIdx.x; 
  uint32_t lane_offset = ((l%16)*n + (8*(l/16)))*sizeof(nv_bfloat16);
  if (l == 0) mbarrier_init(bar,32);
  __syncthreads();
  if (l == 0)
  {
    mbarrier_arrive_expect_tx(bar,m*n*sizeof(nv_bfloat16));
    cp_async_bulk_tensor_2d(As,&a_map, 0,0,bar);
  }
  else mbarrier_arrive(bar); 
  
  mbarrier_wait_parity(bar,0);
  uint32_t ra[4];
  uint32_t sink;
   
  #pragma unroll 
  for (int i = 0; i < iters; i++)
  {
    uint32_t addr = As + compact_swizzle<swizzle_num>(start_offset + lane_offset);

  }
}
#pragma once



namespace tile_sched {

template <int num_tiles, int num_sms>
__device__ __forceinline__
int get_linear_tile_id(int iter, int sm_id)
{
    int tile = (iter * num_sms) + sm_id;

    // Out of range => no tile left
    return (tile < num_tiles) ? tile : -1;
}

__device__ __forceinline__
uint32_t compact1by1(uint32_t x)
{
    x &= 0x55555555u;                 // keep even bits
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

__device__ __forceinline__
void morton_decode_2d(uint32_t tile_id, uint32_t &m, uint32_t &n)
{
    m = compact1by1(tile_id);
    n = compact1by1(tile_id >> 1);
}

  template<
      int group_m,
      int group_n,
      int blocks_per_group,
      int g_outer_m,
      int g_outer_n,
      int BM,
      int BN>
      
  __device__ __forceinline__
  void block_swizzle(
      int b,
      int &block_start_m,
      int &block_start_n
  )
  {
      

      int group_id = b / blocks_per_group;
      int local_id = b % blocks_per_group;


      int global_m = group_id % g_outer_m;
      int global_n = group_id / g_outer_m;


      int local_m = local_id / group_n;
      int local_n = local_id % group_n;


      int tile_m = global_m * group_m + local_m;
      int tile_n = global_n * group_n + local_n;

      block_start_m = tile_m * BM;
      block_start_n = tile_n * BN;
  }


} 

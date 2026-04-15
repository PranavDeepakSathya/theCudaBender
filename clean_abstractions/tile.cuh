#pragma once
#include <stdint.h>

template <int num_tiles, int num_sms>
__device__ __forceinline__
int get_linear_tile_id(int iter, int sm_id)
{
  int tile = (iter * num_sms) + sm_id;
  return (tile < num_tiles) ? tile : -1;
}

__device__ __forceinline__
uint32_t compact1by1(uint32_t x)
{
  x &= 0x55555555u;
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
  int &block_start_n)
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

template<int BBits, int MBase, int SShift, class T>
__device__ __forceinline__
uint32_t cute_swizzle_byte_offset(int elem_offset)
{
  static_assert((SShift >= BBits) || (-SShift >= BBits),
                "abs(SShift) must be >= BBits.");

  uint32_t byte_off = static_cast<uint32_t>(elem_offset * sizeof(T));

  constexpr uint32_t bitmask = (1u << BBits) - 1u;
  constexpr uint32_t yyy_mask =
    bitmask << (MBase + (SShift > 0 ? SShift : 0));

  if constexpr (SShift >= 0) {
    return byte_off ^ ((byte_off & yyy_mask) >> SShift);
  } else {
    return byte_off ^ ((byte_off & yyy_mask) << (-SShift));
  }
}

template<int SwizzleNum>
__device__ __forceinline__
uint32_t compact_swizzle(uint32_t x)
{
  if constexpr (SwizzleNum == 0) {
    return x;
  } else {
    return x ^ ((x & SwizzleNum) >> 3);
  }
}
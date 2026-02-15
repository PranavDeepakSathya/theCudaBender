#pragma once

// Simple persistent tile scheduler
// Maps (sm_id, iter) -> linear tile_id
//
// Guarantees:
//   - no collisions
//   - balanced work across SMs
//   - trivial integer math
//   - GPU-friendly (no modulo, no gcd)
//
// Tile interpretation (tile_id -> (m,n)) is handled separately.

namespace tile_sched {

template <int NUM_TILES, int NUM_SMS>
__device__ __forceinline__
int get_linear_tile_id(int iter, int sm_id)
{
    int tile = (iter * NUM_SMS) + sm_id;

    // Out of range => no tile left
    return (tile < NUM_TILES) ? tile : -1;
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


} 

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

} // namespace tile_sched

#pragma once
#include <stdint.h>

template<int BBits, int MBase, int SShift, class T>
__device__ __forceinline__
uint32_t cute_swizzle_byte_offset(int elem_offset)
{
    static_assert((SShift >= BBits) || (-SShift >= BBits),
                  "abs(SShift) must be >= BBits.");

    // element offset -> byte offset
    uint32_t byte_off =
        static_cast<uint32_t>(elem_offset * sizeof(T));

    // bitmask = (1<<b)-1
    constexpr uint32_t bitmask = (1u << BBits) - 1u;

    // yyy_mask = bitmask << (m + max(0,s))
    constexpr uint32_t yyy_mask =
        bitmask << (MBase + (SShift > 0 ? SShift : 0));

    // Apply cute swizzle in byte space
    if constexpr (SShift >= 0) {
        return byte_off ^ ((byte_off & yyy_mask) >> SShift);
    } else {
        return byte_off ^ ((byte_off & yyy_mask) << (-SShift));
    }
}

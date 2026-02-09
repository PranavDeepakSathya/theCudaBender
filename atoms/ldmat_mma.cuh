#pragma once
#include "utils.cuh"

namespace  warp_atom {

////////////////////////////////////////////////////////////////////////////////
// ldmatrix: A (m8n8.x4)  -> 4 regs
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void ldmatrix_m8n8_x4_b16(
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    const uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

////////////////////////////////////////////////////////////////////////////////
// ldmatrix: B (m8n8.x2)  -> 2 regs
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void ldmatrix_m8n8_x2_b16(
    uint32_t& r0,
    uint32_t& r1,
    const uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr)
    );
}

////////////////////////////////////////////////////////////////////////////////
// mma: m16n8k16 row.col bf16 -> f32 accumulate
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void mma_m16n8k16_row_col_f32_bf16(
    float& c0, float& c1, float& c2, float& c3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

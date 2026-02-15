#pragma once
#include "utils.cuh"

namespace warp_function{

////////////////////////////////////////////////////////////////////////////////
// ldmatrix: A (m8n8.x4) -> reg[4]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void ldmatrix_m8n8_x4_b16(
    uint32_t r[4],
    const uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr)
    );
}

////////////////////////////////////////////////////////////////////////////////
// ldmatrix: B (m8n8.x2) -> reg[2]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void ldmatrix_m8n8_x2_b16(
    uint32_t r[2],
    const uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr)
    );
}

////////////////////////////////////////////////////////////////////////////////
// ldmatrix: (m8n8.x1) -> reg[1]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void ldmatrix_m8n8_x1_b16(
    uint32_t r[1],
    const uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
        "{%0}, [%1];"
        : "=r"(r[0])
        : "r"(smem_addr)
    );
}

////////////////////////////////////////////////////////////////////////////////
// stmatrix: x1 -> reg[1]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void stmatrix_m8n8_x1_b16(
    const uint32_t r[1],
    const uint32_t smem_addr
) {
    asm volatile(
        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};"
        :
        : "r"(smem_addr), "r"(r[0])
    );
}

////////////////////////////////////////////////////////////////////////////////
// stmatrix: x2 -> reg[2]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void stmatrix_m8n8_x2_b16(
    const uint32_t r[2],
    const uint32_t smem_addr
) {
    asm volatile(
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};"
        :
        : "r"(smem_addr), "r"(r[0]), "r"(r[1])
    );
}

////////////////////////////////////////////////////////////////////////////////
// stmatrix: x4 -> reg[4]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void stmatrix_m8n8_x4_b16(
    const uint32_t r[4],
    const uint32_t smem_addr
) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
        :
        : "r"(smem_addr),
          "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
    );
}

////////////////////////////////////////////////////////////////////////////////
// mma: m16n8k16 row.col bf16 -> f32 accumulate
// A: reg[4], B: reg[2], C: f32[4]
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void mma_m16n8k16_row_col_f32_bf16(
    float c[4],
    const uint32_t a[4],
    const uint32_t b[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

} // namespace warp_atom

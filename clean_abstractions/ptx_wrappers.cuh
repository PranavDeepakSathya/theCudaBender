#pragma once
#include "utils.cuh"

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

__device__ __forceinline__
void ldmatrix_m8n8_x4_trans_b16(
    uint32_t r[4],
    const uint32_t smem_addr
) {
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
    : "r"(smem_addr)
  );
}

__device__ __forceinline__
void ldmatrix_m8n8_x2_trans_b16(
    uint32_t r[2],
    const uint32_t smem_addr
) {
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
    "{%0, %1}, [%2];"
    : "=r"(r[0]), "=r"(r[1])
    : "r"(smem_addr)
  );
}

__device__ __forceinline__
void ldmatrix_m8n8_x1_trans_b16(
    uint32_t r[1],
    const uint32_t smem_addr
) {
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
    "{%0}, [%1];"
    : "=r"(r[0])
    : "r"(smem_addr)
  );
}

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

__device__ __forceinline__
void mma_m16n16k16_row_col_f32_bf16(
    float c[8],
    const uint32_t a[4],
    const uint32_t b[4]
) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, "
    "{%8,%9}, "
    "{%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1])
  );
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, "
    "{%8,%9}, "
    "{%0,%1,%2,%3};"
    : "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[2]), "r"(b[3])
  );
}

__device__ __forceinline__
void mma_m16n8k16_row_col_f16_f16(
    uint32_t c[2],
    const uint32_t a[4],
    const uint32_t b[2]
) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0, %1}, "
    "{%2, %3, %4, %5}, "
    "{%6, %7}, "
    "{%0, %1};"
    : "+r"(c[0]), "+r"(c[1])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1])
  );
}

__device__ __forceinline__ uint32_t cvta_smem(void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t addr, int count) {
  asm volatile(
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(addr), "r"(count)
  );
}

__device__ __forceinline__ void mbarrier_arrive(uint32_t addr) {
  asm volatile(
    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
    :: "r"(addr)
    : "memory"
  );
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t addr, int bytes) {
  asm volatile(
    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(addr), "r"(bytes)
    : "memory"
  );
}

__device__ __forceinline__ void mbarrier_wait_parity(uint32_t addr, int phase) {
  asm volatile(
    "{\n\t"
    ".reg .pred P;\n\t"
    "WAIT_LOOP:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
    "P, [%0], %1;\n\t"
    "@!P bra.uni WAIT_LOOP;\n\t"
    "}\n"
    :: "r"(addr), "r"(phase)
    : "memory"
  );
}

__device__ __forceinline__ void tma_fence() {
  asm volatile(
    "fence.proxy.async.shared::cta;\n"
    ::: "memory"
  );
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    uint32_t dst_smem_addr,
    const void* tmap_ptr,
    int x, int y,
    uint32_t mbarrier_addr
) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];\n"
    :: "r"(dst_smem_addr),
       "l"(tmap_ptr),
       "r"(x), "r"(y),
       "r"(mbarrier_addr)
    : "memory"
  );
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_store(
    const void* tmap_ptr,
    int x, int y,
    uint32_t src_smem_addr
) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
    "[%0, {%1, %2}], [%3];\n"
    :: "l"(tmap_ptr),
       "r"(x), "r"(y),
       "r"(src_smem_addr)
    : "memory"
  );
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    uint32_t dst_smem_addr,
    const void* tmap_ptr,
    int x, int y, int z,
    uint32_t mbarrier_addr
) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3, %4}], [%5];\n"
    :
    : "r"(dst_smem_addr),
      "l"(tmap_ptr),
      "r"(x), "r"(y), "r"(z),
      "r"(mbarrier_addr)
    : "memory"
  );
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d_store(
    const void* tmap_ptr,
    int x, int y, int z,
    uint32_t src_smem_addr
) {
  asm volatile(
    "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
    "[%0, {%1, %2, %3}], [%4];\n"
    :
    : "l"(tmap_ptr),
      "r"(x), "r"(y), "r"(z),
      "r"(src_smem_addr)
    : "memory"
  );
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile(
    "cp.async.commit_group;\n"
    ::: "memory"
  );
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile(
    "cp.async.wait_group %0;\n"
    :: "n"(N)
    : "memory"
  );
}
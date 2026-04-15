#pragma once
#include <cuda.h>

__device__ inline uint32_t cvta_smem(void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ inline void mbarrier_init(uint32_t addr, int count) {
  asm volatile(
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(addr), "r"(count)
  );
}

__device__ inline void mbarrier_arrive(uint32_t addr) {
  asm volatile(
    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
    :: "r"(addr)
    : "memory"
  );
}



__device__ inline void mbarrier_arrive_expect_tx(uint32_t addr, int bytes) {
  asm volatile(
    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(addr), "r"(bytes)
    : "memory"
  );
}



__device__ inline void mbarrier_wait_parity(uint32_t addr, int phase) {
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



__device__ inline void tma_fence() {
  asm volatile(
    "fence.proxy.async.shared::cta;\n"
    ::: "memory"
  );
}



__device__ inline void cp_async_bulk_tensor_2d(
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

__device__ inline void cp_async_bulk_tensor_2d_store(
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

__device__ inline void cp_async_bulk_tensor_3d(
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
__device__ inline void cp_async_bulk_tensor_3d_store(
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

__device__ inline void cp_async_commit_group() {
  asm volatile(
    "cp.async.commit_group;\n"
    ::: "memory"
  );
}
template<int N>
__device__ inline void cp_async_wait_group() {
  asm volatile(
    "cp.async.wait_group %0;\n"
    :: "n"(N)
    : "memory"
  );
}



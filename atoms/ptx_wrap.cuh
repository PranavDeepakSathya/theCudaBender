#pragma once
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
// Shared address helper (you already use this pattern)
////////////////////////////////////////////////////////////////////////////////

__device__ inline uint32_t cvta_smem(void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

////////////////////////////////////////////////////////////////////////////////
// mbarrier.init
////////////////////////////////////////////////////////////////////////////////

__device__ inline void mbarrier_init(uint32_t addr, int count) {
  asm volatile(
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(addr), "r"(count)
  );
}

////////////////////////////////////////////////////////////////////////////////
// mbarrier.arrive (release)
////////////////////////////////////////////////////////////////////////////////

__device__ inline void mbarrier_arrive(uint32_t addr) {
  asm volatile(
    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
    :: "r"(addr)
    : "memory"
  );
}

////////////////////////////////////////////////////////////////////////////////
// mbarrier.arrive.expect_tx
// Producer uses this after cp.async.bulk.tensor
////////////////////////////////////////////////////////////////////////////////

__device__ inline void mbarrier_arrive_expect_tx(uint32_t addr, int bytes) {
  asm volatile(
    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(addr), "r"(bytes)
    : "memory"
  );
}

////////////////////////////////////////////////////////////////////////////////
// mbarrier.wait.parity.acquire
// phase is 0/1, computed from reuse count
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Optional: proxy fence for TMA ordering
////////////////////////////////////////////////////////////////////////////////

__device__ inline void tma_fence() {
  asm volatile(
    "fence.proxy.async.shared::cta;\n"
    ::: "memory"
  );
}

////////////////////////////////////////////////////////////////////////////////
// cp.async.bulk.tensor.2d.shared::cta.global
// Your exact form: dst smem addr + tmap + coords + mbarrier
////////////////////////////////////////////////////////////////////////////////

__device__ inline void cp_async_bulk_tensor_2d(
    uint32_t dst_smem_addr,
    const void* tmap_ptr,
    int x, int y,
    uint32_t mbarrier_addr
) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global."
    "mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];\n"
    :: "r"(dst_smem_addr),
       "l"(tmap_ptr),
       "r"(x), "r"(y),
       "r"(mbarrier_addr)
    : "memory"
  );
}


//credits to https://github.com/gau-nernst/learn-cuda/blob/main/02c_matmul_sm120/common.h
//guuci guy. 

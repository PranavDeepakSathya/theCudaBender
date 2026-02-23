#pragma once

template<class Cfg>
struct SmemAllocator {

  static constexpr uint32_t barrier_bytes = 8;

  static constexpr uint32_t A_total =
      Cfg::bk_stages * Cfg::As_bytes;

  static constexpr uint32_t B_total =
      Cfg::bk_stages * Cfg::Bs_bytes;

  static constexpr uint32_t full_total =
      Cfg::bk_stages * barrier_bytes;

  static constexpr uint32_t empty_total =
      Cfg::bk_stages * barrier_bytes;

  static constexpr uint32_t offset_A     = 0;
  static constexpr uint32_t offset_B     = offset_A + A_total;
  static constexpr uint32_t offset_full  = offset_B + B_total;
  static constexpr uint32_t offset_empty = offset_full + full_total;

  static constexpr uint32_t total_bytes =
      offset_empty + empty_total;

  __device__ SmemAllocator(void* smem_raw)
  {
    base_ = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_raw)
    );
  }

  __device__ uint32_t A(int stage) const {
    return base_ + offset_A
         + stage * Cfg::As_bytes;
  }

  __device__ uint32_t B(int stage) const {
    return base_ + offset_B
         + stage * Cfg::Bs_bytes;
  }

  __device__ uint32_t full(int stage) const {
    return base_ + offset_full
         + stage * barrier_bytes;
  }

  __device__ uint32_t empty(int stage) const {
    return base_ + offset_empty
         + stage * barrier_bytes;
  }

private:
  uint32_t base_;
};
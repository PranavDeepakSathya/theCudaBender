#pragma once

template<class Cfg>
struct SmemAllocator
{
  static constexpr uint32_t barrier_bytes = 8;

  //------------------------------------------
  // staged tensor storage
  //------------------------------------------

  static constexpr uint32_t Ks_total =
      Cfg::KVs_stages * Cfg::Ks_bytes;

  static constexpr uint32_t Vs_total =
      Cfg::KVs_stages * Cfg::Vs_bytes;

  //------------------------------------------
  // barrier storage
  //------------------------------------------

  static constexpr uint32_t KV_barriers_total =
      Cfg::KVs_stages * barrier_bytes;

  //------------------------------------------
  // layout
  //------------------------------------------

  // Q aliases Ks[0]
  static constexpr uint32_t offset_Ks = 0;

  static constexpr uint32_t offset_Vs =
      offset_Ks + Ks_total;

  static constexpr uint32_t offset_Q_bar =
      offset_Vs + Vs_total;

  static constexpr uint32_t offset_KV_bars =
      offset_Q_bar + barrier_bytes;

  //------------------------------------------
  // total shared memory
  //------------------------------------------

  static constexpr uint32_t total_bytes =
      offset_KV_bars + KV_barriers_total;

  //------------------------------------------
  // ctor
  //------------------------------------------

  __device__ SmemAllocator(void* smem_raw)
  {
      base_ = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem_raw));
  }

  //------------------------------------------
  // tensors
  //------------------------------------------

  __device__ uint32_t Qs() const
  {
      return base_ + offset_Ks;
  }

  __device__ uint32_t Ks(int stage) const
  {
      return base_ + offset_Ks
           + stage * Cfg::Ks_bytes;
  }

  __device__ uint32_t Vs(int stage) const
  {
      return base_ + offset_Vs
           + stage * Cfg::Vs_bytes;
  }

  //------------------------------------------
  // barriers
  //------------------------------------------

  __device__ uint32_t Q_bar() const
  {
      return base_ + offset_Q_bar;
  }

  __device__ uint32_t KV_bar(int stage) const
  {
      return base_ + offset_KV_bars
           + stage * barrier_bytes;
  }

private:
  uint32_t base_;
};
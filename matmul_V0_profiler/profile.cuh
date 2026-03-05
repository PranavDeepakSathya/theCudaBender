#pragma once
#include <stdint.h>

enum ProfilerTag {
  TAG_SETUP = 0,
  TAG_TMA,
  TAG_LDMATRIX,
  TAG_MMA,
  TAG_STORE
};

__device__ inline uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {

  int64_t* data_ptr_;
  uint32_t sm_id_;
  int cnt_;

  __device__ inline
  void init(int num_entries, int64_t* base_ptr, int warp_stream_id)
  {
    data_ptr_ = base_ptr + warp_stream_id * (1 + num_entries * 4);

    asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id_));

    cnt_ = 0;
  }

  __device__ inline
  void start(int tag)
  {
    int idx = 1 + cnt_ * 4;

    data_ptr_[idx + 0] = sm_id_;
    data_ptr_[idx + 1] = tag;
    data_ptr_[idx + 2] = globaltimer();
  }

  __device__ inline
  void stop()
  {
    int idx = 1 + cnt_ * 4;

    uint64_t end = globaltimer();
    uint64_t start = data_ptr_[idx + 2];

    data_ptr_[idx + 3] = end - start;

    cnt_++;
  }

  __device__ inline
  void flush()
  {
    data_ptr_[0] = cnt_;
  }
};


#pragma once
#include <stdint.h>

// ============================================================
//  Drop-in GPU kernel profiler
//
//  Usage (device side):
//
//    #include "profiler/profile.cuh"
//
//    // define your own tags
//    enum Tag { SETUP=0, TMA, LDMATRIX, MMA, STORE };
//
//    // inside the kernel:
//    Profiler prof;
//    prof.init(NUM_PROF_EVENTS, profiler_ptr, stream_id);
//
//    prof.start(SETUP);
//    /* ... work ... */
//    prof.stop();
//
//    prof.flush();   // once at the end
//
//  Buffer layout per stream:
//    int64[0]              = event count
//    int64[1 + i*4 + 0]   = sm_id
//    int64[1 + i*4 + 1]   = tag
//    int64[1 + i*4 + 2]   = start timestamp (globaltimer ns)
//    int64[1 + i*4 + 3]   = duration (ns)
//
//  Total buffer size:
//    num_streams * (1 + max_events * 4) * sizeof(int64_t)
//
// ============================================================

__device__ __forceinline__
uint64_t gpu_globaltimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
    return t;
}

struct Profiler
{
    int64_t* data_ptr_;
    uint32_t sm_id_;
    int      cnt_;

    // stream_id: unique id for this profiling stream
    //   warp-level:   blockIdx.x * WARPS_PER_BLOCK + warpId
    //   thread-level: blockIdx.x * THREADS_PER_BLOCK + threadIdx.x
    __device__ __forceinline__
    void init(int max_events, int64_t* base_ptr, int stream_id)
    {
        data_ptr_ = base_ptr + (int64_t)stream_id * (1 + max_events * 4);
        asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id_));
        cnt_ = 0;
    }

    __device__ __forceinline__
    void start(int tag)
    {
        int idx    = 1 + cnt_ * 4;
        data_ptr_[idx + 0] = sm_id_;
        data_ptr_[idx + 1] = tag;
        data_ptr_[idx + 2] = gpu_globaltimer();
    }

    __device__ __forceinline__
    void stop()
    {
        uint64_t end   = gpu_globaltimer();
        int      idx   = 1 + cnt_ * 4;
        uint64_t begin = static_cast<uint64_t>(data_ptr_[idx + 2]);
        data_ptr_[idx + 3] = static_cast<int64_t>(end - begin);
        cnt_++;
    }

    __device__ __forceinline__
    void flush()
    {
        data_ptr_[0] = cnt_;
    }
};

// ============================================================
//  Helper: compute buffer size (call from host)
// ============================================================
//
//  auto profiler = torch::zeros(
//      {num_streams, 1 + max_events * 4},
//      torch::TensorOptions().device(dev).dtype(torch::kInt64));
//
// ============================================================

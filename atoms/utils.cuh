#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <type_traits> // for if constexpr
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <numeric>
#include <iomanip>
#include <cudaTypedefs.h>
#include <cuda/barrier>

// 1. The Essential Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// 2. Configuration Enums
enum InitMode {
    MODE_ZEROS, 
    MODE_ARANGE,
    MODE_RAND
};

enum RandomDist {
    DIST_FLOAT_NEG1_1, // [-1.0, 1.0]
    DIST_INT_0_100     // [0, 100]
};

__device__ __forceinline__ bool elect_sync(unsigned int member_mask) {
        uint32_t is_leader;
        asm volatile (
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  elect.sync _|p, %1;\n\t"
            "  selp.u32 %0, 1, 0, p;\n\t"
            "}" 
            : "=r"(is_leader) : "r"(member_mask)
        );
        return (bool)is_leader;
    }



__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}


__device__ __forceinline__ uintptr_t align128(uintptr_t ptr)
{
    return (ptr + 127) & ~uintptr_t(127);
}
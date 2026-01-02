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
#pragma once

#include "utils.cuh"

template <typename T>
class NaiveTensor {
public:
    T* h_ptr = nullptr;
    T* d_ptr = nullptr;
    size_t size;
    size_t bytes;

    // Constructor: Just sets dimensions, doesn't allocate yet (optional style)
    NaiveTensor(size_t N) : size(N), bytes(N * sizeof(T)) {}

    // Destructor: RAII - Clean up automatically when object goes out of scope
    ~NaiveTensor() {
        if (h_ptr) cudaFreeHost(h_ptr);
        if (d_ptr) cudaFree(d_ptr);
    }

    // 1. Allocator
    void allocate() {
        CHECK_CUDA(cudaMallocHost((void**)&h_ptr, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_ptr, bytes));
    }

    // 2. Initializer
    void init_pattern(InitMode mode, RandomDist rand_dist = DIST_FLOAT_NEG1_1) {
        if (!h_ptr) allocate(); // Auto-allocate if forgot

        switch (mode) {
            case MODE_ZEROS:
                memset(h_ptr, 0, bytes);
                break;
            case MODE_ARANGE:
                for(size_t i = 0; i < size; ++i) h_ptr[i] = static_cast<T>(i);
                break;
            case MODE_RAND:
                fill_random(rand_dist);
                break;
        }
    }

    // 3. Sync Methods
    void to_device() {
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
    }

    void to_host() {
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));
    }

    // 4. Smart Printer (Handles the annoying bf16 casting)
    void print_sample(size_t idx, const char* label) {
        // We use a small lambda or cast to ensure printf gets a float
        float val = 0.0f;
        if constexpr (std::is_same<T, nv_bfloat16>::value || std::is_same<T, half>::value) {
            val = static_cast<float>(h_ptr[idx]);
        } else {
            val = static_cast<float>(h_ptr[idx]);
        }
        printf("[%s] Index %lu: %f\n", label, idx, val);
    }

private:
    void fill_random(RandomDist rand_dist) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if (rand_dist == DIST_FLOAT_NEG1_1) {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for(size_t i = 0; i < size; ++i) h_ptr[i] = static_cast<T>(dist(gen));
        } else {
            std::uniform_int_distribution<int> dist(0, 100);
            for(size_t i = 0; i < size; ++i) h_ptr[i] = static_cast<T>(dist(gen));
        }
    }
};
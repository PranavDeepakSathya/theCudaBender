#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <numeric>
#include <iomanip>
#include <cstdarg>

#include "utils.cuh" // Assuming CHECK_CUDA, InitMode, RandomDist are here

enum class Layout {
    ROW_MAJOR,
    COL_MAJOR
};

// Fixed rank for device view to avoid malloc inside kernels
constexpr int MAX_RANK = 8; 

template <typename T>
class NaiveTensor {
public:
    // --------------------------------------------------------
    // Nested: Lightweight View for Device/Kernel Usage
    // --------------------------------------------------------
    struct DeviceView {
        T* ptr;
        size_t strides[MAX_RANK];
        int rank;

        // Device-side accessor
        template<typename... Args>
        __device__ __forceinline__ T& get(Args... indices) const {
            size_t idxs[] = {static_cast<size_t>(indices)...};
            size_t offset = 0;
            #pragma unroll
            for (int i = 0; i < rank; ++i) {
                offset += idxs[i] * strides[i];
            }
            return ptr[offset];
        }

        // Read-only pointer arithmetic for raw access
        __device__ __forceinline__ T* data() const { return ptr; }
    };

    // --------------------------------------------------------
    // Host: Main Class
    // --------------------------------------------------------
    T* h_ptr = nullptr;
    T* d_ptr = nullptr;
    size_t size;
    size_t bytes;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    Layout layout;

    // Updated Constructor: Takes Shape + Layout
    NaiveTensor(std::vector<size_t> shape_in, Layout layout_in = Layout::ROW_MAJOR) 
        : shape(shape_in), layout(layout_in) 
    {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        bytes = size * sizeof(T);
        compute_strides();
    }

    ~NaiveTensor() {
        if (h_ptr) cudaFreeHost(h_ptr);
        if (d_ptr) cudaFree(d_ptr);
    }

    void allocate() {
        CHECK_CUDA(cudaMallocHost((void**)&h_ptr, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_ptr, bytes));
    }

    void init_pattern(InitMode mode, RandomDist rand_dist) {
        if (!h_ptr) allocate();
        switch (mode) {
            case MODE_ZEROS: memset(h_ptr, 0, bytes); break;
            case MODE_ARANGE: for(size_t i=0; i<size; ++i) h_ptr[i] = static_cast<T>(i); break;
            case MODE_RAND: fill_random(rand_dist); break;
        }
    }

    void to_device() { CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice)); }
    void to_host()   { CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost)); }

    // --------------------------------------------------------
    // Host Accessor (CPU)
    // --------------------------------------------------------
    template<typename... Args>
    T& get_host(Args... indices) {
        // Variadic fold expression to verify rank could go here, but keeping it fast.
        size_t idxs[] = {static_cast<size_t>(indices)...};
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            offset += idxs[i] * strides[i];
        }
        return h_ptr[offset];
    }
    template<typename... Args>
    const T& get_host(Args... indices) const {
        // Variadic fold expression to verify rank could go here, but keeping it fast.
        size_t idxs[] = {static_cast<size_t>(indices)...};
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            offset += idxs[i] * strides[i];
        }
        return h_ptr[offset];
    }

    // --------------------------------------------------------
    // Device View Factory
    // --------------------------------------------------------
    DeviceView get_device_view() const {
        DeviceView view;
        view.ptr = d_ptr;
        view.rank = static_cast<int>(shape.size());
        for(int i=0; i<view.rank; ++i) view.strides[i] = strides[i];
        return view;
    }

    // --------------------------------------------------------
    // Utils
    // --------------------------------------------------------
    void pretty_print() {
        to_host();
        std::cout << std::fixed << std::setprecision(4);
        size_t offset = 0;
        print_recursive(0, offset);
        std::cout << std::endl;
    }

private:
    void compute_strides() {
        strides.resize(shape.size());
        if (layout == Layout::ROW_MAJOR) {
            size_t stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        } else { // COL_MAJOR
            size_t stride = 1;
            for (size_t i = 0; i < shape.size(); ++i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
    }

    void fill_random(RandomDist rand_dist) {
        std::random_device rd;
        std::mt19937 gen(rd());
        if (rand_dist == DIST_FLOAT_NEG1_1) {
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            for(size_t i=0; i<size; ++i) h_ptr[i] = static_cast<T>(dist(gen));
        } else {
            std::uniform_int_distribution<int> dist(0, 100);
            for(size_t i=0; i<size; ++i) h_ptr[i] = static_cast<T>(dist(gen));
        }
    }

    void print_recursive(int dim, size_t& offset) {
        // Simplified recursive printer that respects internal shape
        if (dim == shape.size() - 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                float val = static_cast<float>(h_ptr[offset]);
                // Manual stride advance for printing linear buffer sequentially 
                // (Only works if we assume print visits in RowMajor order regardless of layout)
                // For true layout-aware printing, we'd need complex index reconstruction.
                // Assuming "pretty_print" is just for standard debug inspection of the buffer.
                offset++; 
                if (val >= 0) std::cout << " "; 
                std::cout << val;
                if (i < shape[dim] - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else {
            std::cout << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) std::cout << std::string(dim + 1, ' ');
                print_recursive(dim + 1, offset);
                if (i < shape[dim] - 1) std::cout << ",\n";
            }
            std::cout << "]";
        }
    }
};
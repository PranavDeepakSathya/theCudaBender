#include <iostream>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "atoms/all.cuh"

// --- The Atom ---
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t* dst, 
    void* smem_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) 
        : "r"(smem_addr)
    );
}

// --- The Microbenchmark Kernel ---
template <int MODE>
__global__ void smem_microbench(float* out_clocks, uint32_t* dummy_out) {
    // 32 Warps * 16x16 elements * 2 bytes = 16KB
    extern __shared__  nv_bfloat16 smem[];

    int tid = threadIdx.x;
    int wid = tid / 32;

    // 1. Populate SMEM 
    for (int i = tid; i < 32 * 16 * 16; i += blockDim.x) {
        smem[i] = __float2bfloat16((float)i);
    }
    __syncthreads();

    // 2. Determine Read Address
    // Mode 0: Distributed (Warp 0 -> Tile 0, Warp 1 -> Tile 1...)
    // Mode 1: Broadcast (All Warps -> Tile 0)
    int tile_idx = (MODE == 0) ? wid : 0;
    nv_bfloat16* my_tile_ptr = &smem[tile_idx * 16 * 16];

    // Registers
    uint32_t regs[4];
    uint32_t checksum = 0;

    // 3. Measurement Loop
    __syncthreads(); // Align warps before starting clock
    long long start_clock = clock64();
    
    #pragma unroll 16
    for (int i = 0; i < 1024; ++i) {
        ldmatrix_x4(regs, my_tile_ptr);
        
        // Artificial Dependency: XOR loaded data into checksum
        // This forces the compiler to keep the load
        checksum ^= regs[0] ^ regs[1] ^ regs[2] ^ regs[3];
        
        // Conditional pointer bump to prevent overly aggressive loop invariant code motion
        // (Unlikely to be taken, but breaks static analysis)
        if (checksum == 0xFFFFFFFF) my_tile_ptr++; 
    }

    // Fence to ensure clock doesn't slide before the loop finishes
    __threadfence_block(); 
    long long stop_clock = clock64();
    __syncthreads();

    // 4. Report (Thread 0 only)
    if (tid == 0) {
        float total_clocks = (float)(stop_clock - start_clock);
        out_clocks[0] = total_clocks / 1024.0f;
    }

    // 5. Global Sink (Write checksum to memory to prevent dead-code elimination)
    // We write per-thread or per-warp to ensure the work isn't dropped.
    if (tid % 32 == 0) {
        dummy_out[wid] = checksum;
    }
}

int main() {
    int num_warps = 32;
    int num_threads = num_warps * 32;
    
    float* d_out_clocks;
    uint32_t* d_dummy;
    float h_clock;
    
    CHECK_CUDA(cudaMalloc(&d_out_clocks, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dummy, num_warps * sizeof(uint32_t))); // Sink buffer

    int smem_bytes = 32 * 16 * 16 * sizeof(nv_bfloat16); // 16KB

    std::cout << "--- SMEM LdMatrix Latency Benchmark (Fixed) ---" << std::endl;

    // --- CASE 0: DISTRIBUTED ---
    std::cout << "\nRunning DISTRIBUTED (Unique Addresses)..." << std::endl;
    smem_microbench<0><<<1, num_threads, smem_bytes>>>(d_out_clocks, d_dummy);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_clock, d_out_clocks, sizeof(float), cudaMemcpyDeviceToHost));
    
    float lat_dist = h_clock;
    std::cout << "Avg Clocks/Iter: " << lat_dist << std::endl;

    // --- CASE 1: BROADCAST ---
    std::cout << "\nRunning BROADCAST (Same Address)..." << std::endl;
    smem_microbench<1><<<1, num_threads, smem_bytes>>>(d_out_clocks, d_dummy);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_clock, d_out_clocks, sizeof(float), cudaMemcpyDeviceToHost));
    
    float lat_broad = h_clock;
    std::cout << "Avg Clocks/Iter: " << lat_broad << std::endl;

    // --- CONCLUSION ---
    printf("\nRatio (Broadcast/Distributed): %.2fx\n", lat_broad / lat_dist);
    
    if (lat_broad > lat_dist * 1.05) 
        printf("CONCLUSION: Broadcast creates contention/serialization.\n");
    else 
        printf("CONCLUSION: SMEM Broadcast is effectively free.\n");

    cudaFree(d_out_clocks);
    cudaFree(d_dummy);
    return 0;
}
#include "../atoms/all.cuh"

// ============================================================
// LDMATRIX LATENCY (pointer-chased)
// ============================================================
__global__ void smem_latency(uint64_t* out, uint32_t* sink, int iters)
{
    extern __shared__ __align__(128) nv_bfloat16 smem[];

    // --------------------------------------
    // fill smem (single thread is fine)
    // --------------------------------------
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 4096; ++i)
            smem[i] = __float2bfloat16((float)i);
    }

    __syncthreads();

    // --------------------------------------
    // setup
    // --------------------------------------
    uint32_t base =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    uint32_t addr = base;
    uint32_t lane_off = 8 * threadIdx.x * sizeof(nv_bfloat16);

    uint32_t frag[4];

    // --------------------------------------
    // timing
    // --------------------------------------
    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(frag[0]), "=r"(frag[1]),
              "=r"(frag[2]), "=r"(frag[3])
            : "r"(addr + lane_off)
        );

        // ----------------------------------
        // pointer chase
        // ----------------------------------
        uint32_t step = (frag[0] & 0xF) * 32;
        addr = base + step;
    }

    uint64_t end = clock64();

    // --------------------------------------
    // prevent dead-code elimination
    // --------------------------------------
    if (threadIdx.x == 0)
    {
        out[0]  = end - start;
        sink[0] = frag[0];
    }
}


// ============================================================
// HOST BENCHMARK HARNESS
// ============================================================
void run_ldmatrix_latency_bench(
    int iters,
    size_t shared_bytes
)
{
    NaiveTensor<uint64_t> timing({1}, Layout::ROW_MAJOR);
    NaiveTensor<uint32_t> sink({1}, Layout::ROW_MAJOR);

    timing.allocate();
    sink.allocate();

    timing.to_device();
    sink.to_device();

    constexpr int block_size = 32;
    constexpr int grid_size  = 1;

    NaiveLauncher launcher(
        grid_size,
        1,
        block_size,
        shared_bytes
    );

    printf("\n================ LDMATRIX LATENCY ===================\n");
    printf("iters                : %d\n", iters);
    printf("ldmatrix per iter     : 1\n");
    printf("shared memory bytes   : %zu\n\n", shared_bytes);

    // --------------------------------------------------
    // warmup
    // --------------------------------------------------
    for (int i = 0; i < 5; ++i)
    {
        launcher.launch(
            smem_latency,
            timing.d_ptr,
            sink.d_ptr,
            iters
        );
    }

    cudaDeviceSynchronize();

    // --------------------------------------------------
    // benchmark
    // --------------------------------------------------
    launcher.launch(
        smem_latency,
        timing.d_ptr,
        sink.d_ptr,
        iters
    );

    cudaDeviceSynchronize();
    timing.to_host();

    uint64_t cycles = timing.get_host(0);

    double per_ld =
        double(cycles) / double(iters);

    printf("[latency]\n");
    printf("  total cycles        : %llu\n",
           (unsigned long long)cycles);
    printf("  cycles / ldmatrix   : %.2f\n",
           per_ld);

    printf("=====================================================\n\n");
}


// ============================================================
// MAIN
// ============================================================
int main()
{
    int iters = 100000;

    size_t shared_bytes =
        4096 * sizeof(nv_bfloat16);


    run_ldmatrix_latency_bench(iters, shared_bytes);
}

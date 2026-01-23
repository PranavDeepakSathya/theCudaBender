#include "../atoms/all.cuh"

__global__ void mma_latency(uint64_t* out, uint32_t* sink, int iters)
{
    // one warp only
    int lane = threadIdx.x;

    // ----------------------------------
    // fragments
    // ----------------------------------
    uint32_t a[4];   // A fragment
    uint32_t b[2];   // B fragment
    float    c[4];   // C accumulator (m16n8)

    // dummy init
    #pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = 0.0f;

    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, "
            "{%8,%9}, "
            "{%0,%1,%2,%3};"
            :
              "+f"(c[0]), "+f"(c[1]),
              "+f"(c[2]), "+f"(c[3])
            :
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );
    }

    uint64_t end = clock64();

    if (lane == 0)
    {
        out[0] = end - start;
        sink[0] = __float_as_uint(c[0]);
    }
}
__global__ void mma_throughput_8(uint64_t* out, uint32_t* sink, int iters)
{
    int lane = threadIdx.x;

    uint32_t a[4];
    uint32_t b[2];

    float c0[4], c1[4], c2[4], c3[4];
    float c4[4], c5[4], c6[4], c7[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        c0[i]=c1[i]=c2[i]=c3[i]=0.0f;
        c4[i]=c5[i]=c6[i]=c7[i]=0.0f;
    }

    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, "
            "{%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(c0[0]), "+f"(c0[1]), "+f"(c0[2]), "+f"(c0[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, "
            "{%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(c1[0]), "+f"(c1[1]), "+f"(c1[2]), "+f"(c1[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );

        // repeat for c2 ... c7
    }

    uint64_t end = clock64();

    if (lane == 0)
    {
        out[0] = end - start;
        sink[0] = __float_as_uint(c0[0]);
    }
}

void run_mma_latency(int iters)
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
        0
    );

    printf("\n================ MMA LATENCY =================\n");
    printf("iters            : %d\n", iters);
    printf("mma per iter      : 1\n\n");

    // ----------------------------------
    // warmup
    // ----------------------------------
    for (int i = 0; i < 100; ++i)
    {
        launcher.launch(
            mma_latency,
            timing.d_ptr,
            sink.d_ptr,
            iters
        );
        cudaDeviceSynchronize();
    }

    // ----------------------------------
    // benchmark
    // ----------------------------------
    launcher.launch(
        mma_latency,
        timing.d_ptr,
        sink.d_ptr,
        iters
    );

    cudaDeviceSynchronize();
    timing.to_host();

    uint64_t cycles = timing.get_host(0);

    double cycles_per_mma =
        double(cycles) / double(iters);

    printf("[latency]\n");
    printf("  total cycles        : %llu\n",
           (unsigned long long)cycles);
    printf("  cycles / mma        : %.2f\n",
           cycles_per_mma);

    printf("==============================================\n\n");
}

void run_mma_throughput_8(int iters)
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
        0
    );

    printf("\n============= MMA THROUGHPUT (8 ACC) =============\n");
    printf("iters            : %d\n", iters);
    printf("mma per iter      : 8\n\n");

    // ----------------------------------
    // warmup
    // ----------------------------------
    for (int i = 0; i < 100; ++i)
    {
        launcher.launch(
            mma_throughput_8,
            timing.d_ptr,
            sink.d_ptr,
            iters
        );
        cudaDeviceSynchronize();
    }

    // ----------------------------------
    // benchmark
    // ----------------------------------
    launcher.launch(
        mma_throughput_8,
        timing.d_ptr,
        sink.d_ptr,
        iters
    );

    cudaDeviceSynchronize();
    timing.to_host();

    uint64_t cycles = timing.get_host(0);

    double cycles_per_mma =
        double(cycles) / double(iters * 8);

    printf("[throughput]\n");
    printf("  total cycles        : %llu\n",
           (unsigned long long)cycles);
    printf("  cycles / mma        : %.2f\n",
           cycles_per_mma);

    printf("=================================================\n\n");
}

int main()
{
    int iters = 100000;

    run_mma_latency(iters);

    run_mma_throughput_8(iters);

    return 0;
}

#include "../atoms/all.cuh"

__global__ void smem_throughput_8(
    uint64_t* out,
    uint32_t* sink,
    int iters
)
{
    extern __shared__ __align__(128) nv_bfloat16 smem[];

    int lane = threadIdx.x;

    // --------------------------------------
    // init smem
    // --------------------------------------
    for (int i = lane; i < 4096; i += 32)
        smem[i] = __float2bfloat16((float)i);

    __syncthreads();

    uint32_t base =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    uint32_t lane_off = 8 * lane * sizeof(nv_bfloat16);

    // --------------------------------------
    // 8 independent tiles
    // --------------------------------------
    uint32_t a0 = base + 0    + lane_off;
    uint32_t a1 = base + 512  + lane_off;
    uint32_t a2 = base + 1024 + lane_off;
    uint32_t a3 = base + 1536 + lane_off;
    uint32_t a4 = base + 2048 + lane_off;
    uint32_t a5 = base + 2560 + lane_off;
    uint32_t a6 = base + 3072 + lane_off;
    uint32_t a7 = base + 3584 + lane_off;

    uint32_t r0[4], r1[4], r2[4], r3[4];
    uint32_t r4[4], r5[4], r6[4], r7[4];

    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r0[0]), "=r"(r0[1]),
              "=r"(r0[2]), "=r"(r0[3])
            : "r"(a0)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r1[0]), "=r"(r1[1]),
              "=r"(r1[2]), "=r"(r1[3])
            : "r"(a1)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r2[0]), "=r"(r2[1]),
              "=r"(r2[2]), "=r"(r2[3])
            : "r"(a2)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r3[0]), "=r"(r3[1]),
              "=r"(r3[2]), "=r"(r3[3])
            : "r"(a3)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r4[0]), "=r"(r4[1]),
              "=r"(r4[2]), "=r"(r4[3])
            : "r"(a4)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r5[0]), "=r"(r5[1]),
              "=r"(r5[2]), "=r"(r5[3])
            : "r"(a5)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r6[0]), "=r"(r6[1]),
              "=r"(r6[2]), "=r"(r6[3])
            : "r"(a6)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r7[0]), "=r"(r7[1]),
              "=r"(r7[2]), "=r"(r7[3])
            : "r"(a7)
        );
    }

    uint64_t end = clock64();

    if (lane == 0)
    {
        out[0] = end - start;
        sink[0] =
            r0[0] + r1[0] + r2[0] + r3[0] +
            r4[0] + r5[0] + r6[0] + r7[0];
    }
}


void run_ldmatrix_throughput_8(int iters, size_t shared_bytes)
{
    NaiveTensor<uint64_t> timing({1}, Layout::ROW_MAJOR);
    NaiveTensor<uint32_t> sink({1}, Layout::ROW_MAJOR);

    timing.allocate();
    sink.allocate();

    timing.to_device();
    sink.to_device();

    NaiveLauncher launcher(1, 1, 32, shared_bytes);

    // warmup
    for (int i = 0; i < 100; ++i)
    {
        launcher.launch(smem_throughput_8, timing.d_ptr, sink.d_ptr, iters);
           cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();

    launcher.launch(smem_throughput_8, timing.d_ptr, sink.d_ptr, iters);

    cudaDeviceSynchronize();
    timing.to_host();

    double cycles_per_ld =
        double(timing.get_host(0)) / double(iters * 8);

    printf(
        "[throughput 8-frag] cycles / ldmatrix = %.2f\n",
        cycles_per_ld
    );
}


int main()
{
    int iters = 100000;

    size_t shared_bytes =
        8192 * sizeof(nv_bfloat16);

    run_ldmatrix_throughput_8(iters, shared_bytes);

    return 0;
}


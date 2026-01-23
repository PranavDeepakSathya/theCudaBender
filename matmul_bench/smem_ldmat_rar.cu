#include "../atoms/all.cuh"




// ============================================================
// Broadcast: all ldmatrix read SAME smem tile
// ============================================================
__global__ void ldmatrix_broadcast(uint64_t* out, uint32_t* sink, int iters)
{
    extern __shared__ __align__(128) nv_bfloat16 smem[];

    int lane = threadIdx.x;

    // initialize enough shared memory
    for (int i = lane; i < 32 * 128; i += 32)
        smem[i] = __float2bfloat16((float)i * 0.1f);

    __syncthreads();

    uint32_t base =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    uint32_t lane_flat_offset = (8*lane)*sizeof(nv_bfloat16); 

    // precompute addresses OUTSIDE timing loop
    uint32_t a0 = base + 0 + lane_flat_offset;


    uint32_t r0[4];
    uint32_t r1[4];
    uint32_t r2[4];
    uint32_t r3[4];

    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3])
            : "r"(a0)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3])
            : "r"(a0)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r2[0]), "=r"(r2[1]), "=r"(r2[2]), "=r"(r2[3])
            : "r"(a0)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r3[0]), "=r"(r3[1]), "=r"(r3[2]), "=r"(r3[3])
            : "r"(a0)
        );
    }

    uint64_t end = clock64();

    if (lane == 0)
    {
        out[0] = end - start;

        uint32_t acc = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc += r0[i] + r1[i] + r2[i] + r3[i];

        sink[0] = acc;
    }
}


// ============================================================
// Distinct: each ldmatrix reads DIFFERENT smem tile
// ============================================================
__global__ void ldmatrix_distinct(uint64_t* out, uint32_t* sink, int iters)
{
    extern __shared__ __align__(128) nv_bfloat16 smem[];

    int lane = threadIdx.x;

    for (int i = lane; i < 32 * 128; i += 32)
        smem[i] = __float2bfloat16((float)i * 0.01f);

    __syncthreads();

    uint32_t base =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    uint32_t lane_flat_offset = (8*lane)*sizeof(nv_bfloat16); 

    // precompute addresses OUTSIDE timing loop
    uint32_t a0 = base + 0 + lane_flat_offset;
    uint32_t a1 = base + 512 + lane_flat_offset;
    uint32_t a2 = base + 1024 + lane_flat_offset;
    uint32_t a3 = base + 1536 + lane_flat_offset;

    uint32_t r0[4];
    uint32_t r1[4];
    uint32_t r2[4];
    uint32_t r3[4];

    uint64_t start = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3])
            : "r"(a0)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3])
            : "r"(a1)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r2[0]), "=r"(r2[1]), "=r"(r2[2]), "=r"(r2[3])
            : "r"(a2)
        );

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(r3[0]), "=r"(r3[1]), "=r"(r3[2]), "=r"(r3[3])
            : "r"(a3)
        );
    }

    uint64_t end = clock64();

    if (lane == 0)
    {
        out[0] = end - start;

        uint32_t acc = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc += r0[i] + r1[i] + r2[i] + r3[i];

        sink[0] = acc;
    }
}

void run_ldmatrix_microbench(
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

    printf("\n================ LDMATRIX MICROBENCH ================\n");
    printf("iters                : %d\n", iters);
    printf("ldmatrix per iter     : 4\n");
    printf("shared memory bytes   : %zu\n\n", shared_bytes);

    // --------------------------------------------------
    // warmup
    // --------------------------------------------------
    for (int i = 0; i < 5; ++i)
    {
        launcher.launch(
            ldmatrix_broadcast,
            timing.d_ptr,
            sink.d_ptr,
            iters
        );

        launcher.launch(
            ldmatrix_distinct,
            timing.d_ptr,
            sink.d_ptr,
            iters
        );
    }

    cudaDeviceSynchronize();

    // --------------------------------------------------
    // broadcast benchmark
    // --------------------------------------------------
    launcher.launch(
        ldmatrix_broadcast,
        timing.d_ptr,
        sink.d_ptr,
        iters
    );

    cudaDeviceSynchronize();
    timing.to_host();

    uint64_t cycles_broadcast = timing.get_host(0);

    double per_ld_broadcast =
        double(cycles_broadcast) / double(4 * iters);

    printf("[broadcast]\n");
    printf("  total cycles        : %llu\n",
           (unsigned long long)cycles_broadcast);
    printf("  cycles / ldmatrix   : %.2f\n\n",
           per_ld_broadcast);

    // --------------------------------------------------
    // distinct benchmark
    // --------------------------------------------------
    launcher.launch(
        ldmatrix_distinct,
        timing.d_ptr,
        sink.d_ptr,
        iters
    );

    cudaDeviceSynchronize();
    timing.to_host();

    uint64_t cycles_distinct = timing.get_host(0);

    double per_ld_distinct =
        double(cycles_distinct) / double(4 * iters);

    printf("[distinct]\n");
    printf("  total cycles        : %llu\n",
           (unsigned long long)cycles_distinct);
    printf("  cycles / ldmatrix   : %.2f\n",
           per_ld_distinct);

    printf("=====================================================\n\n");
}


int main()
{
    int iters = 100000;

    size_t shared_bytes =
        32 * 128 * sizeof(nv_bfloat16); // matches kernel

    run_ldmatrix_microbench(iters, shared_bytes);
}

/*
SOME THINGS TO THINK ABOUT: 
initially I did not make lane_flat_offset = 8*lane_id*sizeof(bf16)
and I was still seeing about 8.5 to 9 cycles of load lantency per warp. 
in that case, each lane was pointed to the same address, so effectively, 
we were broadcasting the same 8 contigous elements to the (32,4,2) (lane, reg_id, packing_half)
and so ldmatrix seems to broadcast to different registers across the lanes
with no issue. 

Furthermore, the original intention of this was to test INTRA warp broadcast 
across loop iterations, that is, if two different reg fragments request the same 
set of smem addrs for its ldmatrix,in a loop (one after the other)
is there any latency, 
or is it the same as two diff reg fragments request different(dijsoint) set of 
smem addrs for its ldmatrix one after the other? 

The answer is simple, they take the SAME FUCKING TIME (or very close)
*/
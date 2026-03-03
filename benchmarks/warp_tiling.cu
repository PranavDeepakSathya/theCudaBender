#include "../atoms/all.cuh"
namespace wa = warp_function;

static constexpr int mma_m = 16; 
static constexpr int mma_n = 8;
static constexpr int mma_k = 16; 

static constexpr int acc_per_warp_m =8; 
static constexpr int acc_per_warp_n =8; 

static constexpr int WM = mma_m * acc_per_warp_m;
static constexpr int WN = mma_n * acc_per_warp_n;

static constexpr int warps_per_block_m = 2;
static constexpr int warps_per_block_n = 2;

static constexpr int BM = WM*warps_per_block_m; 
static constexpr int BN = WN*warps_per_block_n;
static constexpr int BK = 64;

static constexpr int warp_k_iters = BK / mma_k;
static constexpr int n_bench_iters = 1;   // SET THIS
static constexpr int swizzle_num = 0;
static constexpr CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE;

static constexpr uint32_t shared_bytes =
    (BM * BK + BK * BN + 1024) * sizeof(nv_bfloat16);

static constexpr int block_size =
    warps_per_block_m * warps_per_block_n * 32;

static constexpr int grid_size = 1;


__global__ void warp_tile_bench(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C,
    unsigned long long* clock_diff)
{
    extern __shared__ __align__(1024) uint8_t smem_raw[];

    uint32_t As  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    uint32_t Bs  = As + (BM * BK * sizeof(nv_bfloat16));
    uint32_t bar = Bs + (BK * BN * sizeof(nv_bfloat16));

    int t = threadIdx.x;
    int w = t / 32;
    int l = t % 32;

    int warp_start_m = (w / warps_per_block_n) * WM;
    int warp_start_n = (w % warps_per_block_n) * WN;

    int C_row_start = warp_start_m + (l / 4);
    int C_col_start = warp_start_n + 2 * (l % 4);

    uint32_t a_ld_base =
        ((warp_start_m + (l % 16)) * BK + (8 * (l / 16)))
        * sizeof(nv_bfloat16);

    uint32_t b_ld_base =
        ((warp_start_n + (l % 8)) * BK + (8 * (l / 8)))
        * sizeof(nv_bfloat16);

    float2* C2 = reinterpret_cast<float2*>(C);
    int ldc2 = BN / 2;

    uint32_t ra[acc_per_warp_m][4];
    uint32_t rb[acc_per_warp_n][2];
    float rc[acc_per_warp_m][acc_per_warp_n][4] = {0};

    if (t == 0)
        mbarrier_init(bar, block_size);

    __syncthreads();

    if (t == 0)
    {
        cp_async_bulk_tensor_2d(As, &a_map, 0, 0, bar);
        cp_async_bulk_tensor_2d(Bs, &b_map, 0, 0, bar);
        mbarrier_arrive_expect_tx(
            bar,
            (BM * BK + BK * BN) * sizeof(nv_bfloat16));
    }
    else
    {
        mbarrier_arrive(bar);
    }

    __syncthreads();

    unsigned long long clock_start = 0;
    unsigned long long clock_end   = 0;

    if (t == 0)
        clock_start = clock64();
    __syncthreads();
    
    for (int iter = 0; iter < n_bench_iters; iter++)
    {
        #pragma unroll
        for (int k = 0; k < warp_k_iters; k++)
        {
            #pragma unroll
            for (int m = 0; m < acc_per_warp_m; m++)
            {
                uint32_t a_ld_addr =
                    As + a_ld_base +
                    compact_swizzle<swizzle_num>(((m * mma_m * BK) + (k * mma_k))
                    * sizeof(nv_bfloat16));

                wa::ldmatrix_m8n8_x4_b16(ra[m], a_ld_addr);
            }

            #pragma unroll
            for (int n = 0; n < acc_per_warp_n; n++)
            {
                uint32_t b_ld_addr =
                    Bs + b_ld_base +
                    compact_swizzle<swizzle_num>(((n * mma_n * BK) + (k * mma_k))
                    * sizeof(nv_bfloat16));

                wa::ldmatrix_m8n8_x2_b16(rb[n], b_ld_addr);
            }

            #pragma unroll
            for (int m = 0; m < acc_per_warp_m; m++)
            {
                #pragma unroll
                for (int n = 0; n < acc_per_warp_n; n++)
                {
                    wa::mma_m16n8k16_row_col_f32_bf16(
                        rc[m][n],
                        ra[m],
                        rb[n]);
                }
            }
            rc[0][0][0] += rc[0][0][2];
        }
    }


    __syncthreads();
    if (t == 0)
    {
        clock_end = clock64();
        clock_diff[0] = clock_end - clock_start;
    }


    for (int m = 0; m < acc_per_warp_m; m++)
    {
        #pragma unroll
        for (int n = 0; n < acc_per_warp_n; n++)
        {
            int C_row = C_row_start + (m * mma_m);
            int C_col = (C_col_start + (n * mma_n)) / 2;

            float2 v0 = {rc[m][n][0], rc[m][n][1]};
            float2 v1 = {rc[m][n][2], rc[m][n][3]};

            C2[(C_row) * ldc2 + C_col] = v0;
            C2[(C_row + 8) * ldc2 + C_col] = v1;
        }
    }
}
int main()
{
    constexpr int M = BM;
    constexpr int N = BN;
    constexpr int K = BK;

    NaiveTensor<nv_bfloat16> A({M, K}, Layout::ROW_MAJOR);
    NaiveTensor<nv_bfloat16> B({K, N}, Layout::COL_MAJOR);
    NaiveTensor<float>       C({M, N}, Layout::ROW_MAJOR);

    A.allocate();
    B.allocate();
    C.allocate();

    A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

    A.to_device();
    B.to_device();
    C.to_device();

    CUtensorMap a_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            A.d_ptr, {M, K}, {BM, BK}, swizzle_mode);

    CUtensorMap b_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            B.d_ptr, {K, N}, {BK, BN},swizzle_mode);

    unsigned long long* d_clock;
    unsigned long long  h_clock = 0;

    cudaMalloc(&d_clock, sizeof(unsigned long long));

    NaiveLauncher launcher(
        grid_size,
        1,
        block_size,
        shared_bytes);

    launcher.launch(
        warp_tile_bench,
        a_map,
        b_map,
        C.d_ptr,
        d_clock);

    cudaMemcpy(&h_clock,
               d_clock,
               sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    double flops_per_iter =
    2.0 *
    static_cast<double>(BM) *
    static_cast<double>(BN) *
    static_cast<double>(BK);

    double bytes_per_iter =
        static_cast<double>(BK) *
        static_cast<double>(BM + BN);

    double total_flops =
        flops_per_iter *
        static_cast<double>(n_bench_iters);

    double total_bytes =
        bytes_per_iter *
        static_cast<double>(n_bench_iters);

    double flops_per_cycle =
        total_flops /
        static_cast<double>(h_clock);

    double bytes_per_cycle =
        total_bytes /
        static_cast<double>(h_clock);

    printf("FLOPs per cycle  : %.6f\n", flops_per_cycle);
    printf("Bytes per cycle  : %.6f\n", bytes_per_cycle);
    printf("FLOPs/Bytes per cycle: %.6f\n",flops_per_cycle/bytes_per_cycle);
    printf("mma/ldmatrix: %.6f\n", (double)warps_per_block_m*warps_per_block_n*((acc_per_warp_m*acc_per_warp_n)/(acc_per_warp_m+acc_per_warp_n)));
    
    cudaFree(d_clock);

    return 0;
}
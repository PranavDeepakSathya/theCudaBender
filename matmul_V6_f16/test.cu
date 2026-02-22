#include "../atoms/all.cuh"
#include "config.cuh"
#include "kernel.cuh"
#include <cuda_fp16.h>

namespace ptx = cuda::ptx;

template <class Cfg>
void benchmark_kernel(
    CUtensorMap a_map,
    CUtensorMap b_map,
    CUtensorMap c_map,
    int warmup = 200,
    int iters  = 20000)
{
    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    for (int i = 0; i < warmup; i++)
        launcher.launch(matmul_kernel<Cfg>, a_map, b_map, c_map);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        launcher.launch(matmul_kernel<Cfg>, a_map, b_map, c_map);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_ms = total_ms / iters;
    double avg_s  = avg_ms * 1e-3;

    double flops  = 2.0 * double(Cfg::M) * double(Cfg::N) * double(Cfg::K);
    double tflops = flops / (avg_s * 1e12);

    printf("\n========== Benchmark ==========\n");
    printf("Avg time : %.4f ms\n", avg_ms);
    printf("TFLOP/s  : %.2f\n", tflops);
    printf("================================\n");
}

// ------------------------------------------------------------
// Naive reference (FP16 inputs, FP32 accumulate)
// ------------------------------------------------------------

__global__ void naive_gemm_ref(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++)
    {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[col * K + k]); // col-major B
        acc += a * b;
    }

    C[row * N + col] = __float2half(acc);
}

// ------------------------------------------------------------
// Verification
// ------------------------------------------------------------

void verify_result(
    const NaiveTensor<__half>& C,
    const NaiveTensor<__half>& C_ref,
    int M, int N,
    float abs_thresh = 1e-1f,
    int max_print    = 20)
{
    int bad = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    const float eps = 1e-6f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float ref = __half2float(C_ref.h_ptr[i * N + j]);
            float gpu = __half2float(C.h_ptr[i * N + j]);

            float abs_err = fabsf(ref - gpu);
            float rel_err = abs_err / fmaxf(fabsf(ref), eps);

            max_abs = fmaxf(max_abs, abs_err);
            max_rel = fmaxf(max_rel, rel_err);

            if (abs_err > abs_thresh)
            {
                if (bad < max_print)
                {
                    printf("BAD (%d,%d) ref=%f gpu=%f abs=%e rel=%e\n",
                           i, j, ref, gpu, abs_err, rel_err);
                }
                bad++;
            }
        }
    }

    printf("\n==== Verification ====\n");
    printf("Max abs error : %e\n", max_abs);
    printf("Max rel error : %e\n", max_rel);
    printf("Bad elements  : %d\n", bad);

    if (bad == 0) printf("✅ PASSED\n");
    else          printf("❌ FAILED\n");
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    using Cfg = GemmConfig;

    NaiveTensor<__half> A({Cfg::M, Cfg::K}, Layout::ROW_MAJOR);
    NaiveTensor<__half> B({Cfg::K, Cfg::N}, Layout::COL_MAJOR);

    NaiveTensor<__half> C({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);
    NaiveTensor<__half> C_ref({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);

    A.allocate();
    B.allocate();
    C.allocate();
    C_ref.allocate();

    A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
    C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

    A.to_device();
    B.to_device();
    C.to_device();
    C_ref.to_device();

    CUtensorMap a_map =
        TmaDescriptor<__half>::create_2d_row_major(
            A.d_ptr,
            {Cfg::M, Cfg::K},
            {Cfg::BM, Cfg::BK},
            Cfg::ab_swizzle_mode
        );

    CUtensorMap b_map =
        TmaDescriptor<__half>::create_2d_col_major(
            B.d_ptr,
            {Cfg::K, Cfg::N},
            {Cfg::BK, Cfg::BN},
            Cfg::ab_swizzle_mode
        );

    CUtensorMap c_map =
        TmaDescriptor<__half>::create_2d_row_major(
            C.d_ptr,
            {Cfg::M, Cfg::N},
            {Cfg::BM, Cfg::BN},
            Cfg::c_swizzle_mode
        );

    NaiveLauncher launcher(
        Cfg::grid_size,
        1,
        Cfg::block_size,
        Cfg::shared_bytes
    );

    launcher.launch(matmul_kernel<Cfg>, a_map, b_map, c_map);
    cudaDeviceSynchronize();

    C.to_host();
    printf("Kernel launch complete.\n");

    dim3 block(16, 16);
    dim3 grid(
        (Cfg::N + block.x - 1) / block.x,
        (Cfg::M + block.y - 1) / block.y
    );

    naive_gemm_ref<<<grid, block>>>(
        A.d_ptr,
        B.d_ptr,
        C_ref.d_ptr,
        Cfg::M, Cfg::N, Cfg::K
    );

    cudaDeviceSynchronize();
    C_ref.to_host();

    verify_result(C, C_ref, Cfg::M, Cfg::N);

    benchmark_kernel<Cfg>(a_map, b_map, c_map);

    return 0;
}
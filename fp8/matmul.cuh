#include "../atoms/all.cuh"

typedef __nv_fp8_e4m3 fp8;

constexpr int m = 16;
constexpr int n = 16;   // 2 * 8
constexpr int k = 32;

constexpr int block_size = 32;
constexpr int grid_size  = 1;

__global__ void matmul(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    __grid_constant__ const CUtensorMap c_map
)
{
    extern __shared__ __align__(1024) uint8_t smem_raw[];

    uint32_t smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_raw)
    );

    // Shared memory layout
    uint32_t As  = smem;
    uint32_t Bs  = As + m * k * sizeof(fp8);
    uint32_t bar = Bs + n * k * sizeof(fp8);

    int l = threadIdx.x;

    // Initialize barrier
    if (l == 0) {
        mbarrier_init(bar, block_size);
    }
    __syncthreads();

    // Total bytes expected from TMA
    constexpr uint32_t bytes_expected =
        (m * k + n * k) * sizeof(fp8);

    if (l == 0)
    {
        mbarrier_arrive_expect_tx(bar, bytes_expected);

        cp_async_bulk_tensor_2d(As, &a_map, 0, 0, bar);
        cp_async_bulk_tensor_2d(Bs, &b_map, 0, 0, bar);
    }
    else
    {
        mbarrier_arrive(bar);
    }

    // Wait for phase 0
    mbarrier_try_wait(bar, 0);

    // Register fragments
    uint32_t ra[4];   // A fragment
    uint32_t rb[2][2];   // B fragment
    uint32_t rc[2][2];   // C fragment (m16n16 accumulator)

    
}
#include "../../atoms/all.cuh"

constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 

constexpr int acc_per_warp_m = 2; 
constexpr int acc_per_warp_n = 4; 

constexpr int num_mma_k_iters = 2; 

constexpr int WM = acc_per_warp_m*mma_m; 
constexpr int WN = acc_per_warp_n*mma_n; 

constexpr int warps_per_block_m = 4; 
constexpr int warps_per_block_n = 4; 

constexpr int BM = warps_per_block_m*WM; 
constexpr int BN = warps_per_block_n*WN; 
constexpr int BK = num_mma_k_iters*mma_k;

constexpr int M = 4096;
constexpr int N = 4096; 
constexpr int K = 4096; 

constexpr uint32_t As_bytes = BM*BK*sizeof(nv_bfloat16); 
constexpr uint32_t Bs_bytes = BK*BN*sizeof(nv_bfloat16); 
constexpr uint32_t shared_allocate_bytes = 2*(As_bytes + Bs_bytes + (128*4)); 

constexpr int GM = M/BM; 
constexpr int GN = N/BN;

constexpr int num_BK_iters = K / BK;

constexpr int block_size = warps_per_block_m*warps_per_block_n*32; 
constexpr int grid_size = GM*GN;

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    float* C,
    int M, int N, int K);

__global__ void matmul(__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  float* C)

{
  nv_bfloat16* As[2]; 
  nv_bfloat16* Bs[2]; 
  uint32_t smem_base_a[2];
  uint32_t smem_base_b[2];

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  As[0] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += As_bytes;
  smem = align128(smem);
  As[1] = reinterpret_cast<nv_bfloat16*>(smem); 
  smem += As_bytes;
  smem = align128(smem); 
  Bs[0] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += Bs_bytes; 
  smem = align128(smem); 
  Bs[1] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += Bs_bytes;

  smem_base_a[0] = static_cast<uint32_t>(__cvta_generic_to_shared(As[0]));
  smem_base_a[1] = static_cast<uint32_t>(__cvta_generic_to_shared(As[1]));

  smem_base_b[0] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[0]));
  smem_base_b[1] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[1]));

  uint32_t ra[acc_per_warp_m*4]; //the access is (wm,reg_id) -> 4*wm + reg_id 
  uint32_t rb[acc_per_warp_n*2]; //the access is (wn, reg_id) -> 2*wn + reg_id 
  float rc[acc_per_warp_m*acc_per_warp_n*4] = {0.0f}; //the access is (wm,wn,reg_id) -> acc_per_warp_n*wm*4 + wn*4 + reg_id 

  int l = threadIdx.x % 32; 
  int w = threadIdx.x / 32; 
  int b = blockIdx.x; 

  int warp_start_m = (w / warps_per_block_n)*WM; 
  int warp_start_n = (w % warps_per_block_n)*WN; 

  int block_start_m = (b / GN)*BM; 
  int block_start_n = (b % GN)*BN; 

  __shared__ barrier bar[2]; 



  if (l == 0)
  {
    init(&bar[0],1); 
    init(&bar[1],1); 
  }


  __syncthreads();


  if (is_elected()) {

    int32_t A0[2] = {0 * BK, block_start_m};
    int32_t B0[2] = {0 * BK, block_start_n};

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        As[0], &gA, A0,
        cuda::device::barrier_native_handle(bar[0]));

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        Bs[0], &gB, B0,
        cuda::device::barrier_native_handle(bar[0]));

    cuda::device::barrier_arrive_tx(bar[0], 1, As_bytes + Bs_bytes);


    int32_t A1[2] = {1 * BK, block_start_m};
    int32_t B1[2] = {1 * BK, block_start_n};

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        As[1], &gA, A1,
        cuda::device::barrier_native_handle(bar[1]));

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        Bs[1], &gB, B1,
        cuda::device::barrier_native_handle(bar[1]));

    cuda::device::barrier_arrive_tx(bar[1], 1, As_bytes + Bs_bytes);
  }





  int a_lane_row_base = l % 16;
  int a_lane_col_base = (l / 16) * 8;

  int b_lane_row_base = 8*(l/8);
  int b_lane_col_base = l % 8; 

  int stage = 0;
  uint32_t parity = 0;

  #pragma unroll
  for (int bk = 0; bk < num_BK_iters; ++bk) 
  {
    while (!ptx::mbarrier_try_wait_parity( ptx::sem_acquire, 
      ptx::scope_cta, 
      cuda::device::barrier_native_handle(bar[stage]), parity)) {}

    #pragma unroll
    for (int wk = 0; wk < num_mma_k_iters; wk++)
    {
      #pragma unroll
      for (int wm = 0; wm < acc_per_warp_m; wm++)
      {
        int a_load_shared_row = warp_start_m + (wm*mma_m) + a_lane_row_base;
        int a_load_shared_col = (wk*mma_k) + a_lane_col_base;
        uint32_t a_ld_addr = (smem_base_a[stage]) + ((a_load_shared_col + (BK*a_load_shared_row))*sizeof(nv_bfloat16));
        int a_reg_start = wm*4; 
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(ra[a_reg_start + 0]), "=r"(ra[a_reg_start + 1]), "=r"(ra[a_reg_start + 2]), "=r"(ra[a_reg_start + 3])
          : "r"(a_ld_addr)
        );
      }

      #pragma unroll
      for (int wn = 0; wn < acc_per_warp_n; wn++)
      {
        int b_load_shared_row = (wk*mma_k) + b_lane_row_base; 
        int b_load_shared_col = warp_start_n + (wn*mma_n) + b_lane_col_base;
        uint32_t  b_ld_addr = (smem_base_b[stage]) + ((b_load_shared_row + (BK*b_load_shared_col))*sizeof(nv_bfloat16)); 
        int b_reg_start = wn*2; 
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
          : "=r"(rb[b_reg_start + 0]), "=r"(rb[b_reg_start + 1]) 
          : "r"(b_ld_addr)
        );
      }
      
      #pragma unroll
      for (int wm = 0; wm < acc_per_warp_m; wm++)
      {
        #pragma unroll
        for (int wn = 0; wn < acc_per_warp_n; wn++)
        {
          int a_reg_start = wm*4; 
          int b_reg_start = wn*2; 
          int c_reg_start = (wm*acc_per_warp_n*4) + (wn*4); 
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "=f"(rc[c_reg_start + 0]), "=f"(rc[c_reg_start + 1]), "=f"(rc[c_reg_start + 2]), "=f"(rc[c_reg_start + 3])
            : "r"(ra[a_reg_start + 0]), "r"(ra[a_reg_start + 1]), "r"(ra[a_reg_start + 2]), "r"(ra[a_reg_start + 3]),
              "r"(rb[b_reg_start + 0]), "r"(rb[b_reg_start + 1]),
              "f"(rc[c_reg_start + 0]), "f"(rc[c_reg_start + 1]), "f"(rc[c_reg_start + 2]), "f"(rc[c_reg_start + 3])
          );
        }
      }
      
    }

    __syncthreads();
    int fetch = bk + 2;
    if (fetch < num_BK_iters && is_elected()) {

        int32_t coordsA[2] = {fetch * BK, block_start_m};
        int32_t coordsB[2] = {fetch * BK, block_start_n};

        ptx::cp_async_bulk_tensor(
            ptx::space_shared, ptx::space_global,
            As[stage], &gA, coordsA,
            cuda::device::barrier_native_handle(bar[stage]));

        ptx::cp_async_bulk_tensor(
            ptx::space_shared, ptx::space_global,
            Bs[stage], &gB, coordsB,
            cuda::device::barrier_native_handle(bar[stage]));

        cuda::device::barrier_arrive_tx(
            bar[stage], 1, As_bytes + Bs_bytes);
    }

    // --------------------------------------------------
    // ADVANCE STAGE + PARITY
    // --------------------------------------------------
    stage++;
    if (stage == 2) {
        stage = 0;
        parity ^= 1;
    }
 

 }


  __syncthreads(); 
  //storage: 
  float2* C2 = reinterpret_cast<float2*>(C); 
  int lane_row = l/4; 
  int lane_col = l%4; 
  int ldc2 = N/2; 

  #pragma unroll
  for (int wm = 0; wm < acc_per_warp_m; wm++)
  {
    #pragma unroll
    for (int wn = 0; wn < acc_per_warp_n; wn++)
    {
      int c2_global_row_v0 = block_start_m + warp_start_m + (wm*mma_m) + lane_row + 0;
      int c2_global_row_v1 = c2_global_row_v0 + 8; 
      int c_global_col_elem = 
        block_start_n
        + warp_start_n          // element space
        + (wn * mma_n)
        + (lane_col * 2);         // because float2

      int c2_global_col = c_global_col_elem >> 1;
      //dont worry I shall annotate both this kernels indexing and one_warp_one_mma.cu 
      // and I shall very carefully extract the index maps and tiling isomorphisms. 
      int c_reg_start = ((wn*4) + (wm*acc_per_warp_n*4));
      float2 v0 = {rc[c_reg_start + 0], rc[c_reg_start + 1]}; 
      float2 v1 = {rc[c_reg_start + 2], rc[c_reg_start + 3]}; 
      C2[(c2_global_row_v0)*ldc2 + c2_global_col] = v0; 
      C2[(c2_global_row_v1)*ldc2 + c2_global_col] = v1; 
    }
  }


}

int main()
{
  NaiveTensor<nv_bfloat16> A({M,K}, Layout::ROW_MAJOR);
  NaiveTensor<nv_bfloat16> B({K,N}, Layout::COL_MAJOR); 
  NaiveTensor<float>C({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<float> C_ref({M, N}, Layout::ROW_MAJOR);

  A.allocate();
  B.allocate();
  C.allocate();
  C_ref.allocate(); 

  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  C_ref.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

  A.to_device();
  B.to_device();
  C.to_device();
  C_ref.to_device();
  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{M,K},{BM,BK});
  CUtensorMap b_map = TmaDescriptor<nv_bfloat16>::create_2d_col_major(B.d_ptr, {K,N}, {BK,BN});
  NaiveLauncher launcher(grid_size, 1, block_size,shared_allocate_bytes);

  //correctness launch
  launcher.launch(matmul, a_map, b_map, C.d_ptr);
  cudaDeviceSynchronize(); 
  C.to_host(); 
  

  dim3 block(16, 16);
  dim3 grid(
      (N + block.x - 1) / block.x,
      (M + block.y - 1) / block.y
  );

  naive_gemm_ref<<<grid, block>>>(
      A.d_ptr,
      B.d_ptr,
      C_ref.d_ptr,
      M, N, K
  );

  cudaDeviceSynchronize();

  
  C_ref.to_host();

  const float abs_thresh = 1e0f;
  int bad_count = 0;
  const int max_print = 20000;  // prevent terminal explosion

  float max_rel_err = 0.0f;
  float max_abs_err = 0.0f;

  const float eps = 1e-6f;     // protects divide-by-zero
  const float rel_tol = 1e-2f; // bf16 tensor core tolerance

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
        float ref = C_ref.h_ptr[i * N + j];
        float gpu = C.h_ptr[i * N + j];

        float abs_err = fabsf(ref - gpu);
        float rel_err = abs_err / fmaxf(fabsf(ref), eps);

        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);

        if (abs_err > abs_thresh)
        {
            if (bad_count < max_print)
            {
                printf(
                "BAD (%d,%d) i%%16=%d j%%8=%d i%%32=%d j%%32=%d\n",
                i, j,
                i % 16, j % 8,
                i % 32, j % 32
              );
            }
            bad_count++;
        }
    }
  }

  done:
  printf("Max abs error : %e\n", max_abs_err);
  //printf("Max rel error : %e\n", max_rel_err);
  printf("\nTotal elements with abs error > %.1e : %d\n",
       abs_thresh, bad_count);


printf(" \n ================= C_actual ============ \n");
//C_ref.pretty_print(); 
printf("\n ================ C_computed ============= \n");
//C.pretty_print();


// ============================================================
  // BENCHMARK (AFTER CORRECTNESS IS FULLY VERIFIED)
  // ============================================================

  const int warmup_iters = 20;
  const int bench_iters  = 200;

  // warmup
  for (int i = 0; i < warmup_iters; ++i)
  {
    launcher.launch(matmul, a_map, b_map, C.d_ptr);
  }
  cudaDeviceSynchronize();

  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < bench_iters; ++i)
  {
    launcher.launch(matmul, a_map, b_map, C.d_ptr);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  // average time per kernel in milliseconds
  double avg_ms = ms / bench_iters;

  printf("\n=========================================\n");
  printf("Warmup iters   : %d\n", warmup_iters);
  printf("Benchmark iters: %d\n", bench_iters);
  printf("Average kernel time: %.6f ms\n", avg_ms);
  printf("=========================================\n\n");

  // FLOPs per kernel launch
  double flops = 2.0 * double(M) * double(N) * double(K);

  // convert ms → seconds
  double t_sec = avg_ms * 1e-3;

  // throughput
  double tflops = flops / t_sec / 1e12;

  printf("Problem size: M=%d N=%d K=%d\n", M, N, K);
  printf("Total FLOPs per launch: %.0f\n", flops);
  printf("Throughput: %.2f TFLOP/s\n", tflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Shared memory per block: %u bytes (%.2f KB)\n",
       shared_allocate_bytes,
       shared_allocate_bytes / 1024.0f);


}

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,   // [M, K] row-major
    const nv_bfloat16* B,   // [K, N] column-major
    float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k)
    {
        // A(i,k)
        float a = __bfloat162float(A[row * K + k]);

        // B(k,j) — COLUMN MAJOR
        float b = __bfloat162float(B[col * K + k]);

        acc += a * b;
    }

    C[row * N + col] = acc;
}
#include "../atoms/all.cuh"

constexpr int mma_m = 16; 
constexpr int mma_n = 8;
constexpr int mma_k = 16; 

constexpr int warp_m_acc = 4; 
constexpr int warp_n_acc = 2; 
constexpr int bk_iters = 4;

constexpr int warp_m = 4; 
constexpr int warp_n = 4; 

constexpr int BK = bk_iters*mma_k; 
constexpr int BM = warp_m*mma_m*warp_m_acc; 
constexpr int BN = warp_n*mma_n*warp_n_acc; 


constexpr int K = 4096; 
constexpr int M = BM; 
constexpr int N = BN; 

constexpr int num_block_k_iters = K / BK;

constexpr uint32_t As_bytes = BM*BK*sizeof(nv_bfloat16); 
constexpr uint32_t Bs_bytes = BK*BN*sizeof(nv_bfloat16); 
constexpr uint32_t shared_allocate_bytes = 2*(As_bytes + Bs_bytes + (128*4)); 

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int block_size = warp_m*warp_n*32; 
constexpr int grid_size = 1;

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    float* C,
    int M, int N, int K);

__global__ void one_warp_ilp (__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  float* C)

{
  nv_bfloat16* As[2];
  nv_bfloat16* Bs[2];

  uint32_t smem_base_a[2];
  uint32_t smem_base_b[2];

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  // -----------------------------
  // A tiles
  // -----------------------------
  smem = align128(smem);
  As[0] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += As_bytes;

  smem = align128(smem);
  As[1] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += As_bytes;

  // -----------------------------
  // B tiles
  // -----------------------------
  smem = align128(smem);
  Bs[0] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += Bs_bytes;

  smem = align128(smem);
  Bs[1] = reinterpret_cast<nv_bfloat16*>(smem);
  smem += Bs_bytes;

  // -----------------------------
  // shared addresses for ldmatrix
  // -----------------------------
  smem_base_a[0] = static_cast<uint32_t>(__cvta_generic_to_shared(As[0]));
  smem_base_a[1] = static_cast<uint32_t>(__cvta_generic_to_shared(As[1]));

  smem_base_b[0] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[0]));
  smem_base_b[1] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[1]));

  uint32_t ra[warp_m_acc*4]; 
  uint32_t rb[warp_n_acc*2]; 
  float rc[warp_m_acc*warp_n_acc*4] = {0.0f}; 

  int l = threadIdx.x % 32;
  int w = threadIdx.x / 32;
  int warp_m_start = (w / warp_n)*warp_m_acc*mma_m; 
  int warp_n_start = (w % warp_n)*warp_n_acc*mma_n; 


  __shared__ barrier bar[2]; 

  if (l == 0)
  {
    init(&bar[0],blockDim.x); 
    init(&bar[1],blockDim.x); 
  }


  barrier::arrival_token token[2]; 
  __syncthreads();
  
  if (is_elected())
  {
     

    int32_t coords_A[2] = {0,0};
    int32_t coords_B[2] = {0,0}; 

    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As[0], &gA, coords_A, 
      cuda::device::barrier_native_handle(bar[0]));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs[0], &gB, coords_B, 
      cuda::device::barrier_native_handle(bar[0]));
    token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, As_bytes + Bs_bytes);
  }
  else
  {
    token[0] = bar[0].arrive();
  }

  bar[0].wait(std::move(token[0]));


  int a_lane_row_base = l % 16;
  int a_lane_col_base = (l / 16) * 8;

  int b_lane_row_base = 8 * ((l / 8) % 2);
  int b_lane_col_base = (8 * (l / 16)) + (l % 8);

  for (int bk = 0; bk < num_block_k_iters-1; bk++)
  {
    int curr_stage = bk % 2; 
    int next_stage = (bk+1) % 2; 
    int curr_bk_offset = bk*BK; 
    int next_bk_offset = (bk+1)*BK;

    if (is_elected())
    {
      

      int32_t coords_A[2] = {next_bk_offset,0};
      int32_t coords_B[2] = {next_bk_offset,0}; 

      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As[next_stage], &gA, coords_A, 
        cuda::device::barrier_native_handle(bar[next_stage]));
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs[next_stage], &gB, coords_B, 
        cuda::device::barrier_native_handle(bar[next_stage]));
      token[next_stage] = cuda::device::barrier_arrive_tx(bar[next_stage], 1, As_bytes + Bs_bytes);
    }
    else
    {
      token[next_stage] = bar[next_stage].arrive();
    }


    bar[curr_stage].wait(std::move(token[curr_stage]));
    #pragma unroll
    for (int wk = 0; wk < bk_iters; wk++)
    {
      #pragma unroll
      for (int wm = 0; wm < warp_m_acc; wm ++)
      {
        int a_row = warp_m_start + (wm*mma_m) + a_lane_row_base; 
        int a_col = (wk*mma_k) + a_lane_col_base; 
        int a_ld_addr = smem_base_a[curr_stage] + (a_col + (a_row)*BK)*sizeof(nv_bfloat16);
        int a_reg_start = 4*wm; //a_reg looks like (warp_m_acc,4) in shape 
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(ra[a_reg_start + 0]), "=r"(ra[a_reg_start + 1]), "=r"(ra[a_reg_start + 2]), "=r"(ra[a_reg_start + 3])
          : "r"(a_ld_addr)
        );
      }

      #pragma unroll
      for (int wn = 0; wn < warp_n_acc/2; wn++)
      {
        int b_row = (wk*mma_k) + b_lane_row_base;
        int b_col = warp_n_start + (wn*mma_n*2) + b_lane_col_base;
        int b_ld_addr = smem_base_b[curr_stage] + (b_row + (b_col)*BK)*sizeof(nv_bfloat16);
        int b_reg_start = 4*wn; // double loading b_reg looks like (warp_n_acc, 2) --> (warp_n_acc/2, 4)
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(rb[b_reg_start + 0]), "=r"(rb[b_reg_start + 1]), "=r"(rb[b_reg_start + 2]), "=r"(rb[b_reg_start + 3])
          : "r"(b_ld_addr)
        );
      }
      #pragma unroll
      for (int wm = 0; wm < warp_m_acc; wm ++)
      {
        #pragma unroll
        for (int wn = 0; wn < warp_n_acc; wn++)
        {
          int a_reg_start = 4*wm;
          int b_reg_start = 2*wn; //back to looking at it like (warp_n_acc, 2)
          int c_reg_start = 4*(wn + (wm)*warp_n_acc);
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
  }

  //epilogue
  int bk = num_block_k_iters-1;
  int curr_stage = bk % 2; 


  bar[curr_stage].wait(std::move(token[curr_stage]));
  #pragma unroll
  for (int wk = 0; wk < bk_iters; wk++)
  {
    #pragma unroll
    for (int wm = 0; wm < warp_m_acc; wm ++)
    {
      int a_row = warp_m_start + (wm*mma_m) + a_lane_row_base; 
      int a_col = (wk*mma_k) + a_lane_col_base; 
      int a_ld_addr = smem_base_a[curr_stage] + (a_col + (a_row)*BK)*sizeof(nv_bfloat16);
      int a_reg_start = 4*wm; //a_reg looks like (warp_m_acc,4) in shape 
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(ra[a_reg_start + 0]), "=r"(ra[a_reg_start + 1]), "=r"(ra[a_reg_start + 2]), "=r"(ra[a_reg_start + 3])
        : "r"(a_ld_addr)
      );
    }

    #pragma unroll
    for (int wn = 0; wn < warp_n_acc/2; wn++)
    {
      int b_row = (wk*mma_k) + b_lane_row_base;
      int b_col = warp_n_start + (wn*mma_n*2) + b_lane_col_base;
      int b_ld_addr = smem_base_b[curr_stage] + (b_row + (b_col)*BK)*sizeof(nv_bfloat16);
      int b_reg_start = 4*wn; // double loading b_reg looks like (warp_n_acc, 2) --> (warp_n_acc/2, 4)
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(rb[b_reg_start + 0]), "=r"(rb[b_reg_start + 1]), "=r"(rb[b_reg_start + 2]), "=r"(rb[b_reg_start + 3])
        : "r"(b_ld_addr)
      );
    }
    #pragma unroll
    for (int wm = 0; wm < warp_m_acc; wm ++)
    {
      #pragma unroll
      for (int wn = 0; wn < warp_n_acc; wn++)
      {
        int a_reg_start = 4*wm;
        int b_reg_start = 2*wn; //back to looking at it like (warp_n_acc, 2)
        int c_reg_start = 4*(wn + (wm)*warp_n_acc);
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
  float2* C2 = reinterpret_cast<float2*>(C); 
  int lane_row = l/4; 
  int lane_col = l%4; 
  int ldc2 = N/2; 

  #pragma unroll
  for (int wm = 0; wm < warp_m_acc; wm++)
  {
    #pragma unroll
    for (int wn = 0; wn < warp_n_acc; wn++)
    {
      int c2_global_row_v0 = warp_m_start + (wm*mma_m) + lane_row + 0;
      int c2_global_row_v1 = warp_m_start + (wm*mma_m) + lane_row + 8; 
      int c_global_col_elem =
          warp_n_start          // element space
        + wn * mma_n
        + lane_col * 2;         // because float2

      int c2_global_col = c_global_col_elem >> 1;
      //dont worry I shall annotate both this kernels indexing and one_warp_one_mma.cu 
      // and I shall very carefully extract the index maps and tiling isomorphisms. 
      int c_reg_start = ((wn*4) + (wm*warp_n_acc*4));
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
  launcher.launch(one_warp_ilp, a_map, b_map, C.d_ptr);
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

          if (rel_err > rel_tol)
          {
              printf(
                  "Mismatch at (%d,%d): ref=%f gpu=%f | abs=%e rel=%e\n",
                  i, j, ref, gpu, abs_err, rel_err
              );
              goto done;
          }
      }
  }

  done:
  printf("Max abs error : %e\n", max_abs_err);
  printf("Max rel error : %e\n", max_rel_err);


printf(" \n ================= C_actual ============ \n");
C_ref.pretty_print(); 
printf("\n ================ C_computed ============= \n");
C.pretty_print();


// ============================================================
  // BENCHMARK (AFTER CORRECTNESS IS FULLY VERIFIED)
  // ============================================================

  const int warmup_iters = 20;
  const int bench_iters  = 200;

  // warmup
  for (int i = 0; i < warmup_iters; ++i)
  {
    launcher.launch(one_warp_ilp, a_map, b_map, C.d_ptr);
  }
  cudaDeviceSynchronize();

  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < bench_iters; ++i)
  {
    launcher.launch(one_warp_ilp, a_map, b_map, C.d_ptr);
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
  double gflops = flops / t_sec / 1e9;

  printf("Problem size: M=%d N=%d K=%d\n", M, N, K);
  printf("Total FLOPs per launch: %.0f\n", flops);
  printf("Throughput: %.2f GFLOP/s\n", gflops);

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
#include "../atoms/all.cuh"
constexpr int mma_m = 16; 
constexpr int mma_n = 8;
constexpr int mma_k = 16; 
constexpr int acc_per_warp_m = 2; 
constexpr int acc_per_warp_n = 2; 
constexpr int num_mma_k_iters = 1;
constexpr int WM = mma_m*acc_per_warp_m; 
constexpr int WN = mma_n*acc_per_warp_n;
constexpr int BK = mma_k*num_mma_k_iters;
constexpr int warps_per_block_m = 2; 
constexpr int warps_per_block_n = 2; 
constexpr int BM = warps_per_block_m*WM; 
constexpr int BN = warps_per_block_n*WN; 
constexpr int BM_iters = 2;
constexpr int BN_iters = 2; 
constexpr int block_total_iters = BM_iters*BN_iters; 

constexpr int M = BM_iters*BM; 
constexpr int N = BN_iters*BN; 
constexpr int K = 4096; 


 
constexpr int n_consumer_warps = warps_per_block_m*warps_per_block_n;
constexpr int block_size = ((warps_per_block_m*warps_per_block_n) + (2))*32; 
constexpr int bk_stages = 2;
constexpr int grid_size = 1;

constexpr int prod_warp_id = (n_consumer_warps);
constexpr int p_thread_id = n_consumer_warps*32;
constexpr int num_BK_iters = K / BK; 
constexpr int epilogue_warp_id = n_consumer_warps + 1;
constexpr int ep_thread_id = (n_consumer_warps + 1)*32;

constexpr uint32_t As_bytes = BM*BK*sizeof(nv_bfloat16);
constexpr uint32_t Bs_bytes = BK*BN*sizeof(nv_bfloat16); 
constexpr uint32_t Cs_bytes = BM*BN*sizeof(float); 
constexpr int c_stages = 1;
constexpr uint32_t shared_allocate_bytes = (c_stages*Cs_bytes) + (bk_stages*(As_bytes + Bs_bytes)) + (bk_stages*3*128); 

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

void print_kernel_info();

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    float* C,
    int M, int N, int K);

__global__ void matmul(__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  float* C)

{
  nv_bfloat16* As[bk_stages]; 
  nv_bfloat16* Bs[bk_stages]; 
  uint32_t smem_base_a[bk_stages];
  uint32_t smem_base_b[bk_stages];

  extern __shared__ uint8_t smem_raw[];
  allocate_smem_tiles(smem_raw, As_bytes, Bs_bytes, bk_stages, As, Bs, smem_base_a, smem_base_b); 
  __shared__ barrier full[bk_stages], empty[bk_stages];

  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int warp_start_m = (w / warps_per_block_n)*WM; 
  int warp_start_n = (w % warps_per_block_n)*WN; 

  if (threadIdx.x == 0) {
  for (int i = 0; i < bk_stages; ++i) {
      init(&full[i], (n_consumer_warps*32) + 1);
      init(&empty[i],(n_consumer_warps*32) + 1);
  }
  ptx::fence_proxy_async(ptx::space_shared);
}

  __syncthreads();
  for (int blk_iter = 0; blk_iter < block_total_iters; blk_iter++)
  {
    int blk_iter_m_start = (blk_iter / BN_iters)*BM;
    int blk_iter_n_start = (blk_iter % BN_iters)*BN;
    //producer 
    if (w == prod_warp_id)
    {
      if (t == p_thread_id)
      {
        int stage = 0; 
        #pragma unroll
        for (int bk = 0; bk < num_BK_iters; ++bk, ++stage)
        {
          if (stage ==bk_stages) stage = 0;
          empty[stage].wait(empty[stage].arrive());

          int32_t coordsA[2] = {bk*BK, blk_iter_m_start};
          int32_t coordsB[2] = {bk*BK, blk_iter_n_start};

            ptx::cp_async_bulk_tensor(
                ptx::space_shared, ptx::space_global,
                As[stage], &gA, coordsA,
                cuda::device::barrier_native_handle(full[stage]));

            ptx::cp_async_bulk_tensor(
                ptx::space_shared, ptx::space_global,
                Bs[stage], &gB, coordsB,
                cuda::device::barrier_native_handle(full[stage]));

            barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
                full[stage], 1, As_bytes + Bs_bytes);

        }
      }
    }
    else//consumer logic 
    {
      #pragma unroll
      for (int i = 0; i < bk_stages; ++i) {
            // i initially, all buffers are considered empty; ready for write
            barrier::arrival_token _ = empty[i].arrive();
        }
      
      int stage = 0; 

      int a_lane_row_base = l % 16;
      int a_lane_col_base = (l / 16) * 8;

      int b_lane_row_base = 8*(l/8);
      int b_lane_col_base = l % 8; 

      uint32_t ra[acc_per_warp_m*4]; //the access is (wm,reg_id) -> 4*wm + reg_id 
      uint32_t rb[acc_per_warp_n*2]; //the access is (wn, reg_id) -> 2*wn + reg_id 
      float rc[acc_per_warp_m*acc_per_warp_n*4] = {0.0f}; //the access is (wm,wn,reg_id) -> acc_per_warp_n*wm*4 + wn*4 + reg_id 
      #pragma unroll
      for (int bk = 0; bk < num_BK_iters; ++bk, ++stage)
      {
        if (stage == bk_stages) stage = 0;
        full[stage].wait(full[stage].arrive());
        
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
            warp_atom::ldmatrix_m8n8_x4_b16(ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3], a_ld_addr);

            
          }

          #pragma unroll
          for (int wn = 0; wn < acc_per_warp_n; wn++)
          {
            int b_load_shared_row = (wk*mma_k) + b_lane_row_base; 
            int b_load_shared_col = warp_start_n + (wn*mma_n) + b_lane_col_base;
            uint32_t  b_ld_addr = (smem_base_b[stage]) + ((b_load_shared_row + (BK*b_load_shared_col))*sizeof(nv_bfloat16)); 
            int b_reg_start = wn*2; 
            warp_atom::ldmatrix_m8n8_x2_b16(rb[b_reg_start + 0], rb[b_reg_start + 1],b_ld_addr);
            
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
              warp_atom::mma_m16n8k16_row_col_f32_bf16(
                rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3],
                ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3],
                rb[b_reg_start + 0], rb[b_reg_start + 1],
                rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3]
              );
            }
          }
          
        }
        barrier::arrival_token _ = empty[stage].arrive();
        
      }
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
          int c2_global_row_v0 = blk_iter_m_start + warp_start_m + (wm*mma_m) + lane_row + 0;
          int c2_global_row_v1 = c2_global_row_v0 + 8; 
          int c_global_col_elem =
            blk_iter_n_start 
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
    __syncthreads();
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
  printf("Throughput: %.6f TFLOP/s\n", tflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Shared memory per block: %u bytes (%.2f KB)\n",
       shared_allocate_bytes,
       shared_allocate_bytes / 1024.0f);

  print_kernel_info();

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


void print_kernel_info() {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, matmul);

    if (err != cudaSuccess) {
        printf("cudaFuncGetAttributes failed: %s\n",
               cudaGetErrorString(err));
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n================ cudaFuncGetAttributes ================\n");

    printf("attr.numRegs                     = %d\n",  attr.numRegs);
    printf("attr.sharedSizeBytes             = %zu\n", attr.sharedSizeBytes);
    printf("attr.maxDynamicSharedSizeBytes   = %zu\n", attr.maxDynamicSharedSizeBytes);
    printf("attr.localSizeBytes              = %zu\n", attr.localSizeBytes);
    printf("attr.maxThreadsPerBlock          = %d\n",  attr.maxThreadsPerBlock);
    printf("attr.ptxVersion                  = %d\n",  attr.ptxVersion);
    printf("attr.binaryVersion               = %d\n",  attr.binaryVersion);

    printf("\n================ cudaDeviceProp =======================\n");

    printf("prop.regsPerMultiprocessor       = %d\n",  prop.regsPerMultiprocessor);
    printf("prop.sharedMemPerMultiprocessor  = %zu\n", prop.sharedMemPerMultiprocessor);
    printf("prop.maxThreadsPerMultiProcessor = %d\n",  prop.maxThreadsPerMultiProcessor);
    printf("prop.warpSize                    = %d\n",  prop.warpSize);
    printf("prop.multiProcessorCount         = %d\n",  prop.multiProcessorCount);
    printf("prop.sharedMemPerBlock           = %zu\n", prop.sharedMemPerBlock);
    printf("prop.sharedMemPerBlockOptin      = %zu\n", prop.sharedMemPerBlockOptin);
    printf("prop.maxRegistersPerBlock        = %d\n",  prop.regsPerBlock);

    printf("\n================ derived quantities ===================\n");

    int threads_per_block = block_size;
    int warps_per_block   = threads_per_block / prop.warpSize;

    int regs_per_thread = attr.numRegs;
    int regs_per_block  = regs_per_thread * threads_per_block;

    int max_blocks_regs =
        prop.regsPerMultiprocessor / regs_per_block;

    int max_blocks_threads =
        prop.maxThreadsPerMultiProcessor / threads_per_block;

    size_t smem_per_block =
        attr.sharedSizeBytes + attr.maxDynamicSharedSizeBytes;

    int max_blocks_smem =
        prop.sharedMemPerMultiprocessor / smem_per_block;

    int max_blocks_per_sm =
        min(max_blocks_regs,
            min(max_blocks_threads, max_blocks_smem));

    int active_warps =
        max_blocks_per_sm * warps_per_block;

    int max_warps =
        prop.maxThreadsPerMultiProcessor / prop.warpSize;

    float occupancy =
        float(active_warps) / float(max_warps);

    printf("threads_per_block                = %d\n", threads_per_block);
    printf("warps_per_block                  = %d\n", warps_per_block);
    printf("regs_per_thread                  = %d\n", regs_per_thread);
    printf("regs_per_block                   = %d\n", regs_per_block);
    printf("smem_per_block                   = %zu\n", smem_per_block);
    printf("max_blocks_regs                  = %d\n", max_blocks_regs);
    printf("max_blocks_threads               = %d\n", max_blocks_threads);
    printf("max_blocks_smem                  = %d\n", max_blocks_smem);
    printf("max_blocks_per_sm                = %d\n", max_blocks_per_sm);
    printf("active_warps_per_sm              = %d\n", active_warps);
    printf("max_warps_per_sm                 = %d\n", max_warps);
    printf("theoretical_occupancy            = %.4f\n", occupancy);

    printf("======================================================\n\n");
}

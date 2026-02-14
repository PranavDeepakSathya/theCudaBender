#include "../atoms/all.cuh"

constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 
constexpr int apw_m = 4;
constexpr int apw_n = 4; 
constexpr int wm = mma_m*apw_m; 
constexpr int wn = mma_n*apw_n; 
constexpr int wpb_m = 2;
constexpr int wpb_n = 2; 
constexpr int bm = wpb_m*wm; 
constexpr int bn = wpb_n*wn; 
constexpr int wk_iters = 2;
constexpr int bk = wk_iters*mma_k; 
constexpr int bm_iters = 16;
constexpr int bn_iters = 16; 
constexpr int bk_stages = 2; 
constexpr int producer_id = wpb_m*wpb_n; 
constexpr int block_size = ((wpb_m*wpb_n) + 1)*32;
constexpr int grid_size = 1; 
constexpr int M = bm*bm_iters; 
constexpr int N = bn*bn_iters; 
constexpr int K = 4096;
constexpr int bk_iters = K/bk; 

namespace ptx = cuda::ptx;

constexpr uint32_t As_bytes = bm*bk*sizeof(nv_bfloat16);
constexpr uint32_t Bs_bytes = bk*bn*sizeof(nv_bfloat16);
constexpr uint32_t Cs_bytes = bm*bm*sizeof(float); 
constexpr uint32_t shared_alloc = ((As_bytes+Bs_bytes)*bk_stages) + (Cs_bytes) + (8*1024);

__global__ void matmul(const __grid_constant__ CUtensorMap a_map,
                       const __grid_constant__ CUtensorMap b_map,
                      const __grid_constant__ CUtensorMap c_map)

{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As[bk_stages]; 
  nv_bfloat16* Bs[bk_stages];
  float* Cs;
  uint32_t smem_base_a[bk_stages];
  uint32_t smem_base_b[bk_stages];
  uint32_t smem_base_c;
   
  uint8_t* ptr = smem_raw;

  for (int bk_s = 0; bk_s < bk_stages; bk_s++)
  {
    As[bk_s] = alloc<nv_bfloat16,1024>(ptr, bm*bk);
    smem_base_a[bk_s] = static_cast<uint32_t>(__cvta_generic_to_shared(As[bk_s]));
  }
  for (int bk_s = 0; bk_s < bk_stages; bk_s++)
  {
    Bs[bk_s] = alloc<nv_bfloat16,1024>(ptr, bn*bk);
    smem_base_b[bk_s] = static_cast<uint32_t>(__cvta_generic_to_shared(Bs[bk_s]));
  }

  Cs = alloc<float, 1024>(ptr, bm*bn);
  smem_base_c = static_cast<uint32_t>(__cvta_generic_to_shared(Cs));

  uint64_t* empty = alloc<uint64_t, 8>(ptr, bk_stages);
  uint64_t* full = alloc<uint64_t, 8>(ptr, bk_stages);

  uint64_t empty_tokens[bk_stages];
  uint64_t full_tokens[bk_stages];



  int t = threadIdx.x; 
  int w = t/32; 
  int l = t%32; 
  int a_l_row = l%16;
  int a_l_col = 8*(l/16);
  int b_l_row = 8*(l/8);
  int b_l_col = l%8;
  int warp_m_start = (w/wpb_n)*wm;
  int warp_n_start = (w%wpb_n)*wn; 



  if (t == 0)
  { 
    #pragma unroll
    for (int stage = 0; stage < bk_stages; stage++)
    {
      ptx::mbarrier_init(&empty[stage],(wpb_m*wpb_n + 1)*32);
      ptx::mbarrier_init(&full[stage],(wpb_m*wpb_n + 1)*32);  
    }



  }
  __syncthreads();

  if (w == producer_id)
  {
    for (int block_iter = 0; block_iter < bm_iters*bn_iters; block_iter++)
    {

      int block_m_start = (block_iter/bn_iters)*bm;
      int block_n_start = (block_iter%bn_iters)*bn; 

      for (int bk_idx = 0; bk_idx < bk_iters; bk_idx++)
      {
        int stage = (block_iter*bk_iters + bk_idx) % bk_stages;
        int32_t A_coords[2] = {bk_idx*bk, block_m_start};
        int32_t B_coords[2] = {bk_idx*bk, block_n_start};
        empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]);
        while(!ptx::mbarrier_try_wait(&empty[stage], empty_tokens[stage])); 
        asm volatile("fence.proxy.async.shared::cta;");

        if (l == 0)
        {
          ptx::cp_async_bulk_tensor(
          ptx::space_shared, ptx::space_global,
            As[stage], &a_map, A_coords, &full[stage]);

          ptx::cp_async_bulk_tensor(
          ptx::space_shared, ptx::space_global,
            Bs[stage], &b_map, B_coords, &full[stage]);

          full_tokens[stage] = ptx::mbarrier_arrive_expect_tx(
            ptx::sem_release, 
            ptx::scope_cta, 
            ptx::space_shared,
            &full[stage],
            As_bytes + Bs_bytes
          );
        }
        else
        {
          full_tokens[stage] = ptx::mbarrier_arrive(&full[stage]);
        }
      }
    }
  }

  else
  {

    #pragma unroll
    for (int stage = 0; stage < bk_stages; stage++)
    {
      empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]); 
    }

    for (int block_iter = 0; block_iter < bm_iters*bn_iters; block_iter++)
    {
      uint32_t ra[apw_m*4];
      uint32_t rb[apw_n*2];
      float rc[apw_m*apw_n*4] = {0.0}; 

      
      int block_m_start = (block_iter/bn_iters)*bm;
      int block_n_start = (block_iter%bn_iters)*bn; 

      #pragma unroll
      for (int bk_idx = 0; bk_idx < bk_iters; bk_idx++)
      {
        int stage = (block_iter*bk_iters + bk_idx) % bk_stages;

        full_tokens[stage] = ptx::mbarrier_arrive(&full[stage]);
        while(!ptx::mbarrier_try_wait(&full[stage], full_tokens[stage]));

        #pragma unroll
        for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++)
        {
          #pragma unroll
          for (int wm_idx = 0; wm_idx < apw_m; wm_idx++)
          {
            int a_ld_shared_offset = (warp_m_start + (wm_idx)*mma_m + (l%16))*bk + ((wk_idx)*mma_k + 8*(l/16));
            nv_bfloat16* a_ptr = As[stage] + a_ld_shared_offset;
            uint32_t a_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(a_ptr));
            int a_reg_start = wm_idx*4; 
            warp_atom::ldmatrix_m8n8_x4_b16(ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3], a_smem_addr);

          }
          #pragma unroll
          for (int wn_idx = 0; wn_idx < apw_n; wn_idx++)
          {
            int b_ld_shared_offset = (warp_n_start + (wn_idx)*mma_n + (l%8))*bk+ ((wk_idx)*mma_k + 8*(l/8));
            nv_bfloat16* b_ptr = Bs[stage] + b_ld_shared_offset; 
            uint32_t b_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(b_ptr));
            int b_reg_start = wn_idx*2; 
            warp_atom::ldmatrix_m8n8_x2_b16(rb[b_reg_start + 0], rb[b_reg_start + 1],b_smem_addr);
          }
          #pragma unroll
          for (int wm_idx = 0; wm_idx < apw_m; wm_idx++)
          {
            #pragma unroll
            for (int wn_idx = 0; wn_idx < apw_n; wn_idx++)
            {
              int a_reg_start = wm_idx*4; 
              int b_reg_start = wn_idx*2; 
              int c_reg_start = 4*(wn_idx + apw_n*wm_idx); 
              warp_atom::mma_m16n8k16_row_col_f32_bf16(
                rc[c_reg_start + 0], rc[c_reg_start + 1], rc[c_reg_start + 2], rc[c_reg_start + 3],
                ra[a_reg_start + 0], ra[a_reg_start + 1], ra[a_reg_start + 2], ra[a_reg_start + 3],
                rb[b_reg_start + 0], rb[b_reg_start + 1]

              );
            }
          }

        }

        empty_tokens[stage] = ptx::mbarrier_arrive(&empty[stage]); 
      }

      float2* C2 = reinterpret_cast<float2*>(Cs); 
      int lane_row = l/4; 
      int lane_col = 2*(l%4); 
      int ldc2 = bn/2;
      ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());
      #pragma unroll
      for (int wm_idx = 0; wm_idx < apw_m; wm_idx++)
      {
        #pragma unroll
        for (int wn_idx = 0; wn_idx < apw_n; wn_idx++)
        {
          int C_row = warp_m_start + (wm_idx*mma_m) + lane_row;
          int C_col = (warp_n_start + (wn_idx*mma_n) + lane_col)/2;
          int c_reg_start = 4*(wn_idx + apw_n*wm_idx); 
          float2 v0 = {rc[c_reg_start + 0], rc[c_reg_start + 1]}; 
          float2 v1 = {rc[c_reg_start + 2], rc[c_reg_start + 3]}; 
          C2[(C_row)*ldc2 + (C_col)] = v0; 
          C2[(C_row+8)*ldc2 + (C_col)] = v1;
        }
      }

      asm volatile("fence.proxy.async.shared::cta;");
      sync_bar<wpb_m*wpb_n*32>();
      int32_t C_coords[2] = {block_n_start, block_m_start};
      
      if (l == 0)
      {
        cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_global, cuda::ptx::space_shared,
        &c_map, C_coords, Cs
        );

        cuda::ptx::cp_async_bulk_commit_group();
      }


    }

  }

}

__global__ void naive_gemm_ref(
    const nv_bfloat16* A,   // [M,K] row-major
    const nv_bfloat16* B,   // [K,N] col-major
    float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++)
    {
        float a = __bfloat162float(A[row * K + k]);
        float b = __bfloat162float(B[col * K + k]); // col-major

        acc += a * b;
    }

    C[row * N + col] = acc;
}


// ======================= Verification =======================

void verify_result(
    const NaiveTensor<float>& C,
    const NaiveTensor<float>& C_ref,
    int M, int N,
    float abs_thresh,
    int max_print)
{
    int bad_count = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    const float eps = 1e-6f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
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
                        "BAD (%d,%d) ref=%f gpu=%f abs=%e rel=%e\n",
                        i, j, ref, gpu, abs_err, rel_err
                    );
                }
                bad_count++;
            }
        }
    }

    printf("\n==== Verification ====\n");
    printf("Max abs error : %e\n", max_abs_err);
    printf("Max rel error : %e\n", max_rel_err);
    printf("Bad elements  : %d (abs > %.2e)\n",
           bad_count, abs_thresh);

    if (bad_count == 0)
        printf("✅ PASSED\n");
    else
        printf("❌ FAILED\n");
}


// ======================= Benchmark =======================

void benchmark_matmul(
    const char* name,
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    CUtensorMap c_map,
    int warmup_iters = 10,
    int bench_iters  = 100)
{
    for (int i = 0; i < warmup_iters; i++)
        launcher.launch(matmul, a_map, b_map, c_map);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < bench_iters; i++)
        launcher.launch(matmul, a_map, b_map, c_map);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_ms = total_ms / bench_iters;
    double avg_s  = avg_ms * 1e-3;

    double flops  = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_s * 1e12);

    printf("\n========== Benchmark ==========\n");
    printf("Kernel   : %s\n", name);
    printf("Shape    : M=%d N=%d K=%d\n", M, N, K);
    printf("Avg time : %.4f ms\n", avg_ms);
    printf("TFLOP/s  : %.2f\n", tflops);
    printf("================================\n");
}


// ======================= Print Config =======================

void print_cfg()
{
    printf("WM = %d\n", wm);
    printf("WN = %d\n", wn);
    printf("BK = %d\n", bk);

    printf("BM = %d\n", bm);
    printf("BN = %d\n", bn);

    printf("Compute warps = %d\n", wpb_m * wpb_n);
    printf("Total warps   = %d\n", wpb_m * wpb_n + 1);
    printf("Threads/block = %d\n", block_size);

    printf("bk_iters = %d\n", bk_iters);

    printf("As_bytes = %u\n", As_bytes);
    printf("Bs_bytes = %u\n", Bs_bytes);
    printf("shared_bytes = %u\n", shared_alloc);

    printf("Grid size = %d blocks\n", grid_size);
}


// ======================= Main =======================

int main()
{
    NaiveTensor<nv_bfloat16> A({M, K}, Layout::ROW_MAJOR);
    NaiveTensor<nv_bfloat16> B({K, N}, Layout::COL_MAJOR);
    NaiveTensor<float>       C({M, N}, Layout::ROW_MAJOR);
    NaiveTensor<float>       C_ref({M, N}, Layout::ROW_MAJOR);

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

    // ---- TMA descriptors ----

    CUtensorMap a_map =
        TmaDescriptor<nv_bfloat16>::create_2d_row_major(
            A.d_ptr, {M, K}, {bm, bk});

    CUtensorMap b_map =
        TmaDescriptor<nv_bfloat16>::create_2d_col_major(
            B.d_ptr, {K, N}, {bk, bn});

    
    CUtensorMap c_map =
        TmaDescriptor<float>::create_2d_row_major(
            C.d_ptr, {M, N}, {bm, bn});

    NaiveLauncher launcher(grid_size, 1, block_size, shared_alloc);

    launcher.launch(matmul, a_map, b_map, c_map);

    cudaDeviceSynchronize();
    C.to_host();

    printf("Launch complete.\n");

    // ---- Reference ----

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    naive_gemm_ref<<<grid, block>>>(
        A.d_ptr,
        B.d_ptr,
        C_ref.d_ptr,
        M, N, K);

    cudaDeviceSynchronize();
    C_ref.to_host();

    verify_result(C, C_ref, M, N, 1e-1, 20);

    benchmark_matmul(
        "sm120 bf16 matmul",
        launcher,
        a_map,
        b_map,
        c_map);

    print_cfg();
}
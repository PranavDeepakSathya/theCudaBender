#include "../atoms/all.cuh"

constexpr  int M = 1024*13; 
constexpr  int N = 1024*13;
constexpr  int SM = 1024;
constexpr  int SN = 1024; 
constexpr  int BM = 64; 
constexpr  int BN = 128; //32*32 = 1024 elements = 2048 bytes, make sure we dont fuck aligment up 
//by making this smaller than 1024 bytes. 

constexpr int num_stages = 4;
constexpr uint32_t shared_bytes = (num_stages*BM*BN*sizeof(nv_bfloat16)) + 4*1024; 
constexpr uint32_t tile_size = BM*BN*sizeof(nv_bfloat16);
constexpr int BM_iters = SM/BM; 
constexpr  int BN_iters = SN/BN; 
constexpr int total_iters = BM_iters*BN_iters; 
constexpr int GM = M/SM;
constexpr int GN = N/SN; 
constexpr int N_blocks_per_grid = GM*GN; 



__global__ void copy_kern(__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap a_out_map)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  nv_bfloat16* As[num_stages]; 
  uint8_t* ptr = smem_raw;
  int t = threadIdx.x; 
  int w = t/32;
  int l = t % 32; 
  int b = blockIdx.x; 
  int b_start_m = (b/GN)*SM; 
  int b_start_n = (b%GN)*SN;

  for (int s = 0; s < num_stages; s++)
  {
    As[s] = alloc<nv_bfloat16, 1024>(ptr, BM*BN);
  }
  uint64_t* empty = alloc<uint64_t, 8>(ptr, num_stages);
  uint64_t* full = alloc<uint64_t, 8>(ptr, num_stages);
  uint64_t empty_tokens[num_stages];
  uint64_t full_tokens[num_stages]; 


  if (t == 0)
  {
    for (int s = 0; s < num_stages; s++)
    {
      cuda::ptx::mbarrier_init(&empty[s],64);
      cuda::ptx::mbarrier_init(&full[s],64);

    }
  }
  __syncthreads();

  if (w == 0) //producer code 
  {
    for (int idx = 0; idx < total_iters; idx ++)
    {
      int stage = idx % num_stages; 
      int32_t coords[2] = {(idx % BN_iters)*BN + b_start_n, (idx / BN_iters)*BM + b_start_m};
      
      empty_tokens[stage] = cuda::ptx::mbarrier_arrive(&empty[stage]);
      while(!cuda::ptx::mbarrier_try_wait(&empty[stage], empty_tokens[stage])); 

      if (l == 0)
      {

        

        cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_shared, cuda::ptx::space_global,
          As[stage], &a_map, coords, &full[stage]);
        
        full_tokens[stage] = cuda::ptx::mbarrier_arrive_expect_tx(
          cuda::ptx::sem_release, 
          cuda::ptx::scope_cta, 
          cuda::ptx::space_shared,
          &full[stage],
          BM*BN*sizeof(nv_bfloat16)
        );
        
      }
      else
      {
         full_tokens[stage] = cuda::ptx::mbarrier_arrive(&full[stage]);
      }

    }
  }
  else
  {
    for (int s = 0; s < num_stages; s++)
    {
      empty_tokens[s] = cuda::ptx::mbarrier_arrive(&empty[s]);
    }

    for (int idx = 0; idx < total_iters; idx ++)
    {
      int stage = idx % num_stages; 
      int32_t coords[2] = {(idx % BN_iters)*BN + b_start_n, (idx / BN_iters)*BM + b_start_m};
      full_tokens[stage] = cuda::ptx::mbarrier_arrive(&full[stage]);
      while(!cuda::ptx::mbarrier_try_wait(&full[stage], full_tokens[stage]));

      if(l == 0)
      {
        cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_global, cuda::ptx::space_shared,
        &a_out_map, coords, As[stage]
      );

      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
      
      empty_tokens[stage] = cuda::ptx::mbarrier_arrive(&empty[stage]);
      }
      else
      {
        empty_tokens[stage] = cuda::ptx::mbarrier_arrive(&empty[stage]);
      }

    }
  }
}

int main()
{
  NaiveTensor<nv_bfloat16>A({M,N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>A_out({M,N},Layout::ROW_MAJOR); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A.to_device();
  A_out.to_device(); 

  CUtensorMap a_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A.d_ptr,{M,N},{BM,BN});
  CUtensorMap a_out_map = TmaDescriptor<nv_bfloat16>::create_2d_row_major(A_out.d_ptr,{M,N},{BM,BN});

  NaiveLauncher launcher(N_blocks_per_grid, 1, 64,shared_bytes);
  launcher.launch(copy_kern, a_map, a_out_map);
  cudaDeviceSynchronize(); 
  A_out.to_host(); 
  
  printf("\n ====================A================\n");
  //A.pretty_print(); 
  printf("\n ====================A_out ================\n");
  //A_out.pretty_print(); 

    // ---- CPU-side equality check ----
  bool ok = true;
  int first_bad = -1;

  auto *Ah  = reinterpret_cast<const uint16_t*>(A.h_ptr);
  auto *Aoh = reinterpret_cast<const uint16_t*>(A_out.h_ptr);

  for (int i = 0; i < M * N; i++) {
    if (Ah[i] != Aoh[i]) {
      ok = false;
      first_bad = i;
      break;
    }
  }

  if (ok) {
    printf("✅ Copy correctness: PASSED (A == A_out)\n");
  } else {
    int r = first_bad / N;
    int c = first_bad % N;
    printf("❌ Copy correctness: FAILED at idx=%d (row=%d col=%d)\n",
          first_bad, r, c);
    printf("   A     bits = 0x%04x\n", Ah[first_bad]);
    printf("   A_out bits = 0x%04x\n", Aoh[first_bad]);
  }

  // ---- Throughput benchmark ----
  constexpr int iters = 2000;

  // warmup
  for (int i = 0; i < 50; i++) {
    launcher.launch(copy_kern, a_map, a_out_map);
  }
  cudaDeviceSynchronize();

  // timing with CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < iters; i++) {
    launcher.launch(copy_kern, a_map, a_out_map);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  // bytes moved per kernel
  double bytes_per_copy = double(M) * double(N) * sizeof(nv_bfloat16);

  // load + store traffic
  double bytes_per_kernel = 2.0 * bytes_per_copy;

  double total_bytes = bytes_per_kernel * iters;

  // GB/s
  double seconds = ms / 1e3;
  double gb = total_bytes / 1e9;
  double gbps = gb / seconds;

  printf("\n==== Benchmark ====\n");
  printf("Matrix size: %d x %d bf16\n", M, N);
  printf("Bytes per kernel (load+store): %.3f MB\n", bytes_per_kernel / 1e6);
  printf("Iterations: %d\n", iters);
  printf("Time: %.3f ms total\n", ms);
  printf("Throughput: %.2f GB/s\n", gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);


}
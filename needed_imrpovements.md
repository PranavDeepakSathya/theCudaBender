we are hitting 345 tflops even after swizzling and persistance: the following are the possible culprits
(all in context of rtx 6000 pro)
1. way too many device functions? not sure about this one, this is the least of our concerns

2. we need to get register counts. after applying swizzling, wk_iters = 4 is the fastest. 
Therefore, if we have enough regs avaiable, we can allocate regs such that all of the wk_iters 
can be loaded into rmem at once. Indeed my old trick was to use ldmatrix x4 and load 2 tiles of b_smem at once
across wk_iters, gau.naurst does the same thing 
  auto compute = [&](int stage_id) {
    // A smem->rmem
    for (int m = 0; m < WARP_M / MMA_M; m++)
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        int addr = A_smem_thread + stage_id * AB_size;
        addr += m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_rmem[m][k], addr ^ (k * 32));
      }

    // B smem->rmem
    for (int n = 0; n < WARP_N / MMA_N; n++)
      for (int k = 0; k < BLOCK_K / MMA_K; k += 2) {
        int addr = B_smem_thread + stage_id * AB_size;
        addr += n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_rmem[n][k], addr ^ (k * 32));
      }

    // MMA
    for (int m = 0; m < WARP_M / MMA_M; m++)
      for (int n = 0; n < WARP_N / MMA_N; n++)
        for (int k = 0; k < BLOCK_K / MMA_K; k++)
          mma_m16n8k16(A_rmem[m][k], B_rmem[n][k], acc[m][n]);


https://github.com/gau-nernst/learn-cuda/blob/main/02c_matmul_sm120/matmul_v1.cu

3. general register useage and pollution cleanup is needed, clean refactors of kernels and utilities is needed
,essentially we need to make space for those extra regs. 

4. In leu of persistance, we need to get our cudaFuncAttributes thingy and ensure we are large enough to schedule 
exactly 1 block per sm, and we need to cleanup many of our loops

5. finally we need to try different threadblock swizzles (we are using morton order rn) and try to hit max on sm_120. 
6. last but not least, since bk_stages = 2 is the only sane one, we can move from tokens to parity hopefully that reduces some overhead. 


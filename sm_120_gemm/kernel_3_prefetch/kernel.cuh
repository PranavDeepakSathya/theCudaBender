#pragma once
#include "../gemm_maker.cuh"

template <class Cfg>
__global__ void matmul_kernel(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float* C
)
{
  GemmMaker<Cfg> gemm(a_map, b_map, threadIdx.x);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  int b = blockIdx.x;
  int t = threadIdx.x;
  int l = t % 32;
  int w = t / 32;
  int block_start_m;
  int block_start_n;
  block_swizzle<Cfg::group_m, Cfg::group_n, Cfg::blocks_per_group,
                Cfg::G_outer_M, Cfg::G_outer_N, Cfg::BM, Cfg::BN>
               (b, block_start_m, block_start_n);
  float2* C2 = reinterpret_cast<float2*>(C);

  uint32_t As_base   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  uint32_t Bs_base   = As_base + Cfg::As_bytes * Cfg::bk_stages;
  uint32_t full_base = Bs_base + Cfg::Bs_bytes * Cfg::bk_stages;

  auto As       = [&](int s) { return As_base   + s * Cfg::As_bytes; };
  auto Bs       = [&](int s) { return Bs_base   + s * Cfg::Bs_bytes; };
  auto full_bar = [&](int s) { return full_base + s * 8; };

  uint32_t ra[Cfg::wk_stages][Cfg::acc_per_warp_m][4];
  uint32_t rb[Cfg::wk_stages][Cfg::acc_per_warp_n][2];
  float rc[Cfg::acc_per_warp_m][Cfg::acc_per_warp_n][4] = {0.0};

  // init barriers
  if (t == 0) {
    for (int s = 0; s < Cfg::bk_stages; s++) mbarrier_init(full_bar(s), 32);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  __syncthreads();

  // prime outer bk pipeline: fire all bk_stages TMAs upfront
  auto issue_tma = [&](int bk_idx, int stage) {
    if (l == 0) {
      mbarrier_arrive_expect_tx(full_bar(stage), Cfg::As_bytes + Cfg::Bs_bytes);
      gemm.load_A_g2s(bk_idx, block_start_m, As(stage), full_bar(stage));
      gemm.load_B_g2s(bk_idx, block_start_n, Bs(stage), full_bar(stage));
    } else {
      mbarrier_arrive(full_bar(stage));
    }
  };

  #pragma unroll
  for (int s = 0; s < Cfg::bk_stages; s++) {
    if (w == 0) issue_tma(s, s);
  }
  mbarrier_wait_parity(full_bar(0), 0);

  // prime inner wk pipeline on bk stage 0: (wk_stages - 1) ldmatrix load
  #pragma unroll
  for (int i = 0; i < Cfg::wk_stages - 1; i++) {
    gemm.load_A_s2r(ra[i], As(0), i);
    gemm.load_B_s2r(rb[i], Bs(0), i);
  }

  static constexpr int full_bk_iters = Cfg::block_k_iters - Cfg::bk_stages;
  static constexpr int wk_iters      = Cfg::warp_k_iters  - (Cfg::wk_stages - 1);

  // steady state: TMA + wk-pipeline fused, wk crosses bk boundary 
  for (int bk_idx = 0; bk_idx < full_bk_iters; bk_idx++) {
    int bk_cons_stage      = bk_idx % Cfg::bk_stages;
    int next_bk_cons_stage = (bk_idx + 1) % Cfg::bk_stages;
    int parity             = ((bk_idx + 1) / Cfg::bk_stages) % 2;
    int next_bk_load_idx   = bk_idx + Cfg::bk_stages;
    int next_bk_load_stage = next_bk_load_idx % Cfg::bk_stages;
    int bk_base            = bk_idx * Cfg::warp_k_iters;

    // phase 1: inner wk-pipe on current bk_cons_stage
    for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++) {
      int wk_load_idx     = (wk_idx + (Cfg::wk_stages - 1)) % Cfg::warp_k_iters;
      int wk_load_stage   = (bk_base + wk_load_idx) % Cfg::wk_stages;
      int wk_compute_stage= (bk_base + wk_idx)      % Cfg::wk_stages;

      gemm.load_A_s2r(ra[wk_load_stage], As(bk_cons_stage), wk_load_idx);
      gemm.load_B_s2r(rb[wk_load_stage], Bs(bk_cons_stage), wk_load_idx);
      gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
    }

    __syncthreads();                                   // loads of curr bk done
    if (w == 0) issue_tma(next_bk_load_idx, next_bk_load_stage);
    mbarrier_wait_parity(full_bar(next_bk_cons_stage), parity);
    __syncthreads();

    // phase 2: inner wk-pipe crosses onto next_bk_cons_stage
    for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++) {
      int wk_load_idx     = (wk_idx + (Cfg::wk_stages - 1)) % Cfg::warp_k_iters;
      int wk_load_stage   = (bk_base + wk_load_idx) % Cfg::wk_stages;
      int wk_compute_stage= (bk_base + wk_idx)      % Cfg::wk_stages;

      gemm.load_A_s2r(ra[wk_load_stage], As(next_bk_cons_stage), wk_load_idx);
      gemm.load_B_s2r(rb[wk_load_stage], Bs(next_bk_cons_stage), wk_load_idx);
      gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
    }
  }

  //drain: same structure as steady state, no more TMAs issued 
  static constexpr int no_tma_end = full_bk_iters + (Cfg::bk_stages - 1);

  for (int bk_idx = full_bk_iters; bk_idx < no_tma_end; bk_idx++) {
    int bk_cons_stage      = bk_idx % Cfg::bk_stages;
    int next_bk_cons_stage = (bk_idx + 1) % Cfg::bk_stages;
    int parity             = ((bk_idx + 1) / Cfg::bk_stages) % 2;
    int bk_base            = bk_idx * Cfg::warp_k_iters;

    for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++) {
      int wk_load_idx     = (wk_idx + (Cfg::wk_stages - 1)) % Cfg::warp_k_iters;
      int wk_load_stage   = (bk_base + wk_load_idx) % Cfg::wk_stages;
      int wk_compute_stage= (bk_base + wk_idx)      % Cfg::wk_stages;

      gemm.load_A_s2r(ra[wk_load_stage], As(bk_cons_stage), wk_load_idx);
      gemm.load_B_s2r(rb[wk_load_stage], Bs(bk_cons_stage), wk_load_idx);
      gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
    }

    __syncthreads();
    mbarrier_wait_parity(full_bar(next_bk_cons_stage), parity);
    __syncthreads();

    for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++) {
      int wk_load_idx     = (wk_idx + (Cfg::wk_stages - 1)) % Cfg::warp_k_iters;
      int wk_load_stage   = (bk_base + wk_load_idx) % Cfg::wk_stages;
      int wk_compute_stage= (bk_base + wk_idx)      % Cfg::wk_stages;

      gemm.load_A_s2r(ra[wk_load_stage], As(next_bk_cons_stage), wk_load_idx);
      gemm.load_B_s2r(rb[wk_load_stage], Bs(next_bk_cons_stage), wk_load_idx);
      gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
    }
  }

  //  epilogue: last bk stage, no next bk to cross into 
  static constexpr int bk_idx         = Cfg::block_k_iters - 1;
  static constexpr int bk_cons_stage  = bk_idx % Cfg::bk_stages;
  static constexpr int bk_base        = bk_idx * Cfg::warp_k_iters;

  for (int wk_idx = 0; wk_idx < wk_iters; wk_idx++) {
    int wk_load_idx     = (wk_idx + (Cfg::wk_stages - 1)) % Cfg::warp_k_iters;
    int wk_load_stage   = (bk_base + wk_load_idx) % Cfg::wk_stages;
    int wk_compute_stage= (bk_base + wk_idx)      % Cfg::wk_stages;

    gemm.load_A_s2r(ra[wk_load_stage], As(bk_cons_stage), wk_load_idx);
    gemm.load_B_s2r(rb[wk_load_stage], Bs(bk_cons_stage), wk_load_idx);
    gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
  }
  __syncthreads();

  for (int wk_idx = wk_iters; wk_idx < Cfg::warp_k_iters; wk_idx++) {
    int wk_compute_stage = (bk_base + wk_idx) % Cfg::wk_stages;
    gemm.mma(rc, ra[wk_compute_stage], rb[wk_compute_stage]);
  }

  __syncthreads();   // absolutely needed before store
  gemm.store_C(C2, rc, block_start_m, block_start_n);
}

template <class Cfg>
inline void launch_matmul(
    NaiveLauncher& launcher,
    CUtensorMap a_map,
    CUtensorMap b_map,
    float* C_dev
)
{
  launcher.launch(matmul_kernel<Cfg>, a_map, b_map, C_dev);
}
struct GemmInputs {
  static constexpr int warp_m_tiles = 4;
  static constexpr int warp_n_tiles = 4;
  static constexpr int block_m_warps = 4;
  static constexpr int block_n_warps = 4;
  static constexpr int k_stages = 2;
  static constexpr int problem_m = 4096;
  static constexpr int problem_n = 4096;
  static constexpr int problem_k = 4096;
  static constexpr int bk_mma_slices = 2;
};


template <class In>
struct GemmConfig {
  // ===============================
  // inputs (re-exported)
  // ===============================
  using Inputs = In;

  static constexpr int warp_m_tiles  = Inputs::warp_m_tiles;
  static constexpr int warp_n_tiles  = Inputs::warp_n_tiles;
  static constexpr int block_m_warps = Inputs::block_m_warps;
  static constexpr int block_n_warps = Inputs::block_n_warps;
  static constexpr int k_stages      = Inputs::k_stages;
  static constexpr int problem_m     = Inputs::problem_m;
  static constexpr int problem_n     = Inputs::problem_n;
  static constexpr int problem_k     = Inputs::problem_k;
  static constexpr int bk_mma_slices = Inputs::bk_mma_slices;

  // ===============================
  // instruction atom
  // ===============================
  static constexpr int mma_m = 16;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;

  // ===============================
  // warp tiles
  // ===============================
  static constexpr int WM = mma_m * warp_m_tiles;
  static constexpr int WN = mma_n * warp_n_tiles;

  // ===============================
  // block tiles
  // ===============================
  static constexpr int BM = block_m_warps * WM;
  static constexpr int BN = block_n_warps * WN;

  // ===============================
  // k tiling
  // ===============================

  static constexpr int BK = mma_k * num_mma_k_iters;

  // ===============================
  // grid
  // ===============================
  static constexpr int GM = problem_m / BM;
  static constexpr int GN = problem_n / BN;

  // ===============================
  // execution
  // ===============================
  static constexpr int n_consumer_warps =
      block_m_warps * block_n_warps;

  static constexpr int n_producer_warps = 1;

  static constexpr int block_threads =
      (n_consumer_warps + n_producer_warps) * 32;

  static constexpr int num_k_iters =
      problem_k / BK;

  // ===============================
  // memory
  // ===============================
  static constexpr uint32_t As_bytes =
      BM * BK * sizeof(nv_bfloat16);

  static constexpr uint32_t Bs_bytes =
      BK * BN * sizeof(nv_bfloat16);

  static constexpr uint32_t smem_bytes =
      k_stages * (As_bytes + Bs_bytes);

  // ===============================
  // invariants
  // ===============================

  // exact tiling
  static_assert(problem_m % BM == 0,
                "M must be divisible by BM");
  static_assert(problem_n % BN == 0,
                "N must be divisible by BN");
  static_assert(problem_k % BK == 0,
                "K must be divisible by BK");

  // warp legality
  static_assert(n_consumer_warps + n_producer_warps <= 32,
                "too many warps per block");

  // thread legality
  static_assert(block_threads <= 1024,
                "block exceeds max threads");

  // pipeline sanity
  static_assert(num_k_iters >= k_stages,
                "not enough K-iterations to fill pipeline");

  // shared memory (100 KB hard limit)
  static_assert(smem_bytes < 100 * 1024,
                "shared memory usage too large");
};

using Cfg = GemmConfig<GemmInputs>;

#include "utils.cuh"
#include "sm120_bf16f32acc_config.cuh"
#include "ldmat_mma.cuh"

template <class Cfg>
struct GemmProblem {
  // host tensors
  NaiveTensor<nv_bfloat16> A;
  NaiveTensor<nv_bfloat16> B;
  NaiveTensor<float>       C;

  // tensor maps
  CUtensorMap tmaA;
  CUtensorMap tmaB;

};

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

template <class Cfg>
GemmProblem<Cfg> make_gemm_problem() {
  using AType = nv_bfloat16;
  using BType = nv_bfloat16;
  using CType = float;

  constexpr int M = Cfg::problem_m;
  constexpr int N = Cfg::problem_n;
  constexpr int K = Cfg::problem_k;

  // --------------------------------------------------
  // 1. Construct tensors (layouts FORCED here)
  // --------------------------------------------------
  GemmProblem<Cfg> prob{
    // A: [M, K] row-major
    NaiveTensor<AType>({M, K}, Layout::ROW_MAJOR),

    // B: [K, N] col-major
    NaiveTensor<BType>({K, N}, Layout::COL_MAJOR),

    // C: [M, N] row-major
    NaiveTensor<CType>({M, N}, Layout::ROW_MAJOR)
  };

  // allocate + initialize
  prob.A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  prob.B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  prob.C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

  prob.A.to_device();
  prob.B.to_device();
  prob.C.to_device();

  // --------------------------------------------------
  // 2. Create TMA descriptors (derived from Cfg)
  // --------------------------------------------------

  // A: row-major, tiles [BM, BK]
  prob.tmaA =
    TmaDescriptor<AType>::create_2d_row_major(
      prob.A.d_ptr,
      {M, K},
      {Cfg::BM, Cfg::BK}
    );

  // B: col-major, tiles [BK, BN]
  prob.tmaB =
    TmaDescriptor<BType>::create_2d_col_major(
      prob.B.d_ptr,
      {K, N},
      {Cfg::BK, Cfg::BN}
    );


  return prob;
}


template<class Cfg>
struct TmaLoadA {

  __device__ static void run(
      CUtensorMap const& tmaA,
      void* smem_dst,
      barrier* bar,
      int b,
      int k_iter
  ) {
    int block_m = b / Cfg::GN;

    int32_t coord[2] = {
      block_m * Cfg::BM,
      k_iter  * Cfg::BK
    };

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        smem_dst, &tmaA, coord,
        cuda::device::barrier_native_handle(bar));
  }
};



template<class Cfg>
struct TmaLoadB {

  __device__ static void run(
      CUtensorMap const& tmaB,
      void* smem_dst,
      barrier* bar,
      int b,
      int k_iter
  ) {
    int block_n = b % Cfg::GN;

    int32_t coord[2] = {
      k_iter  * Cfg::BK,
      block_n * Cfg::BN
    };

    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global,
        smem_dst, &tmaB, coord,
        cuda::device::barrier_native_handle(bar));
  }
};



template<class Cfg>
struct CStore {

  __device__ static void run(
      float* gC,      // global C base
      int b,          // linear CTA index
      int w,          // consumer warp id
      int l,          // consumer lane id
      float* accum    // warp accumulator fragment
  ) {

    int m_start =
        (b / Cfg::GN) * Cfg::BM +
        (w / Cfg::block_n_warps) * Cfg::WM;

    int n_start =
        (b % Cfg::GN) * Cfg::BN +
        (w % Cfg::block_n_warps) * Cfg::WN;

    int lane_m = l / 4;
    int lane_n = 2 * (l % 4);

    float2* C2 = reinterpret_cast<float2*>(gC);
    int ldc2   = Cfg::problem_n / 2;

    for (int wm_iter = 0; wm_iter < Cfg::warp_m_tiles; wm_iter++) {
      for (int wn_iter = 0; wn_iter < Cfg::warp_n_tiles; wn_iter++) {

        int c2_row =
            m_start + wm_iter * Cfg::mma_m + lane_m;

        int c2_col =
            (n_start + wn_iter * Cfg::mma_n + lane_n) / 2;

        int c_reg_start =
            (4 * wn_iter) +
            (4 * Cfg::warp_n_tiles * wm_iter);

        float2 v0 = make_float2(
            accum[c_reg_start + 0],
            accum[c_reg_start + 1]);

        float2 v1 = make_float2(
            accum[c_reg_start + 2],
            accum[c_reg_start + 3]);

        C2[c2_row * ldc2 + c2_col] = v0;
        C2[(c2_row + 8) * ldc2 + c2_col] = v1;
      }
    }
  }
};


template<class Cfg>
struct LdMatrixA {

  __device__ static void run(
      uint32_t regA[Cfg::warp_m_tiles * 4],
      uint32_t smem_base_a,
      int wk_iter,
      int w,
      int l
  ) 
  {
    
    #pragma unroll
    for (int wm_iter = 0; wm_iter < Cfg::warp_m_tiles; wm_iter++) {

      
      int m = (w / Cfg::block_n_warps) * Cfg::WM + (wm_iter)*Cfg::mma_m + (l % 16);
      int k = (wk_iter)*Cfg::mma_k + ((l / 16) * 8);
      uint32_t smem_addr = smem_base_a + ((m*Cfg::BK + k)*sizeof(nv_bfloat16));
      int a_reg_start = wm_iter*4;
      warp_atom::ldmatrix_m8n8_x4_b16(regA[a_reg_start + 0], regA[a_reg_start + 1], regA[a_reg_start + 2], regA[a_reg_start + 3], smem_addr);
     
    }
  }
};


template<class Cfg>
struct LdMatrixB {

  __device__ static void run(
      uint32_t regB[Cfg::warp_n_tiles * 2],
      uint32_t smem_base_b,
      int wk_iter,
      int w,
      int l
  ) 
  {
    
    #pragma unroll
    for (int wn_iter = 0; wn_iter < Cfg::warp_n_tiles; wn_iter++) {

      
      int n = (w % Cfg::block_n_warps) * Cfg::WN + (wn_iter) * Cfg::mma_n + (l % 8);
      int k = (wk_iter)*Cfg::mma_k + (8*(l/8));
      uint32_t smem_addr = smem_base_b + ((n*Cfg::BK + k)*sizeof(nv_bfloat16));
      int b_reg_start = wn_iter*2; 
      warp_atom::ldmatrix_m8n8_x2_b16(regB[b_reg_start + 0], regB[b_reg_start + 1],smem_addr);
     
    }
  }
};  


template<class Cfg>
struct MmaLoop {

  __device__ static void run(
      float regC[Cfg::warp_m_tiles * Cfg::warp_n_tiles * 4],
      uint32_t regA[Cfg::warp_m_tiles * 4],
      uint32_t regB[Cfg::warp_n_tiles * 2]
  ) {

    #pragma unroll
    for (int wm_iter = 0; wm_iter < Cfg::warp_m_tiles; wm_iter++) {

      int a_reg_start = wm_iter * 4;

      #pragma unroll
      for (int wn_iter = 0; wn_iter < Cfg::warp_n_tiles; wn_iter++) {

        int b_reg_start = wn_iter * 2;

        int c_reg_start =
            (wm_iter * Cfg::warp_n_tiles + wn_iter) * 4;

        warp_atom::mma_m16n8k16_row_col_f32_bf16(
            regC[c_reg_start + 0],
            regC[c_reg_start + 1],
            regC[c_reg_start + 2],
            regC[c_reg_start + 3],

            regA[a_reg_start + 0],
            regA[a_reg_start + 1],
            regA[a_reg_start + 2],
            regA[a_reg_start + 3],

            regB[b_reg_start + 0],
            regB[b_reg_start + 1]

        );
      }
    }
  }
};

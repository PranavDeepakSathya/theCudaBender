#include "utils.cuh"
#include "sm120_bf16f32acc_config.cuh"

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

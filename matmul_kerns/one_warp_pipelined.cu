#include "../atoms/all.cuh"
constexpr int mma_m = 16; 
constexpr int mma_n = 8; 
constexpr int mma_k = 16; 

constexpr int BK_stages = 2; //there thins ARE FIXED, cant change em ever 
//cause we are declaring static named regs. 
constexpr int acc_per_warp_m = 2; 
constexpr int acc_per_warp_n = 4; 

constexpr int BM = mma_m*acc_per_warp_m; 
constexpr int BN = mma_n*acc_per_warp_n; 
constexpr int BK_stage_iters = 16; 
constexpr int BK = mma_k*BK_stages*BK_stage_iters; 

constexpr size_t As_bytes = BM * BK * sizeof(nv_bfloat16);
constexpr size_t Bs_bytes = BK * BN * sizeof(nv_bfloat16); 
constexpr size_t shared_allocate_bytes = As_bytes + Bs_bytes + (4*128);


__global__ void warp_matmul (__grid_constant__ const CUtensorMap gA, 
  __grid_constant__ const CUtensorMap gB,
  NaiveTensor<float>::DeviceView C)

{

  extern __shared__ uint8_t smem_raw[];
  uintptr_t smem = reinterpret_cast<uintptr_t>(smem_raw);

  smem = align128(smem);
  nv_bfloat16* As = (nv_bfloat16*)smem;
  smem += BM * BK * sizeof(nv_bfloat16);
  smem = align128(smem);
  nv_bfloat16* Bs = (nv_bfloat16*)smem;

  uint32_t smem_base_a = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t smem_base_b = static_cast<uint32_t>(__cvta_generic_to_shared(Bs));

    __shared__ barrier bar; 

  if (l == 0)
  {
    init(&bar, threads_per_block); 
  }

  __syncthreads();

  barrier::arrival_token token; 
  
  if (is_elected())
  {
     

    int32_t coords_A[2] = {0,0};
    int32_t coords_B[2] = {0,0}; 

    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, As, &gA, coords_A, cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, Bs, &gB, coords_B, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, As_bytes + Bs_bytes);
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); 

  // we shall use the format ra_{bk_stage_id}_{acc_per_m_id}_{reg_id}
  //rb_{bk_stage_id}_{acc_per_n_id}_{reg_id}
  //rc_{acc_per_m_id}_{acc_per_n_id}_{reg_id}

  // stage 0
  uint32_t ra_0_0_0, ra_0_0_1, ra_0_0_2, ra_0_0_3;
  uint32_t ra_0_1_0, ra_0_1_1, ra_0_1_2, ra_0_1_3;

  // stage 1
  uint32_t ra_1_0_0, ra_1_0_1, ra_1_0_2, ra_1_0_3;
  uint32_t ra_1_1_0, ra_1_1_1, ra_1_1_2, ra_1_1_3;

  // stage 0
  uint32_t rb_0_0_0, rb_0_0_1;
  uint32_t rb_0_1_0, rb_0_1_1;
  uint32_t rb_0_2_0, rb_0_2_1;
  uint32_t rb_0_3_0, rb_0_3_1;

  // stage 1
  uint32_t rb_1_0_0, rb_1_0_1;
  uint32_t rb_1_1_0, rb_1_1_1;
  uint32_t rb_1_2_0, rb_1_2_1;
  uint32_t rb_1_3_0, rb_1_3_1;

  // m = 0
  float rc_0_0_0 = 0.0f, rc_0_0_1 = 0.0f, rc_0_0_2 = 0.0f, rc_0_0_3 = 0.0f;
  float rc_0_1_0 = 0.0f, rc_0_1_1 = 0.0f, rc_0_1_2 = 0.0f, rc_0_1_3 = 0.0f;
  float rc_0_2_0 = 0.0f, rc_0_2_1 = 0.0f, rc_0_2_2 = 0.0f, rc_0_2_3 = 0.0f;
  float rc_0_3_0 = 0.0f, rc_0_3_1 = 0.0f, rc_0_3_2 = 0.0f, rc_0_3_3 = 0.0f;

  // m = 1
  float rc_1_0_0 = 0.0f, rc_1_0_1 = 0.0f, rc_1_0_2 = 0.0f, rc_1_0_3 = 0.0f;
  float rc_1_1_0 = 0.0f, rc_1_1_1 = 0.0f, rc_1_1_2 = 0.0f, rc_1_1_3 = 0.0f;
  float rc_1_2_0 = 0.0f, rc_1_2_1 = 0.0f, rc_1_2_2 = 0.0f, rc_1_2_3 = 0.0f;
  float rc_1_3_0 = 0.0f, rc_1_3_1 = 0.0f, rc_1_3_2 = 0.0f, rc_1_3_3 = 0.0f;

  // int b_lane_group_16x8_id = l/8; 
  // int b_lane_col_id = l % 8; 
  // int b_lane_group_offset = b_lane_group_16x8_id*8; 
  // int b_col_idx = warp_n_start + b_lane_col_id; 


  // int a_lane_group_16x16_id = l/16; 
  // int a_lane_row_id = l % 16;
  // int a_lane_group_offset = a_lane_group_16x16_id*8; 
  // int a_row_idx = warp_m_start + a_lane_row_id; 

  int a_lane_row = l % 16;
  int a_lane_col_base = (l / 16) * 8;

  int b_lane_row_base = (l / 8) * 8;
  int b_lane_col = l % 8;

  // warp tile geometry
  int A_row_base[acc_per_warp_m] = {
      a_lane_row,
      a_lane_row + mma_m
  };

  int B_col_base[acc_per_warp_n] = {
      b_lane_col,
      b_lane_col + mma_n,
      b_lane_col + 2 * mma_n,
      b_lane_col + 3 * mma_n
  };

  for (int wk = 0; wk < BK; wk +=mma_k*BK_stages)
  {
    uint32_t la_0_0 =
      smem_base_a +
      ((A_row_base[0] * BK + (a_lane_col_base + wk)) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ra_0_0_0), "=r"(ra_0_0_1),
          "=r"(ra_0_0_2), "=r"(ra_0_0_3)
        : "r"(la_0_0)
    );

    // stage 0, m = 1
    uint32_t la_0_1 =
        smem_base_a +
        ((A_row_base[1] * BK + (a_lane_col_base + wk)) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ra_0_1_0), "=r"(ra_0_1_1),
          "=r"(ra_0_1_2), "=r"(ra_0_1_3)
        : "r"(la_0_1)
    );

    // stage 1, m = 0
    uint32_t la_1_0 =
        smem_base_a +
        ((A_row_base[0] * BK + (a_lane_col_base + wk + mma_k)) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ra_1_0_0), "=r"(ra_1_0_1),
          "=r"(ra_1_0_2), "=r"(ra_1_0_3)
        : "r"(la_1_0)
    );

    // stage 1, m = 1
    uint32_t la_1_1 =
        smem_base_a +
        ((A_row_base[1] * BK + (a_lane_col_base + wk + mma_k)) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ra_1_1_0), "=r"(ra_1_1_1),
          "=r"(ra_1_1_2), "=r"(ra_1_1_3)
        : "r"(la_1_1)
    );
    
    uint32_t lb_0_and_1_0 =
    smem_base_b +
    (((b_lane_row_base + wk) + (B_col_base[0])*BK) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(rb_0_0_0), "=r"(rb_0_0_1),
          "=r"(rb_1_0_0), "=r"(rb_1_0_1)
        : "r"(lb_0_and_1_0)
    );

    uint32_t lb_0_and_1_1 =
    smem_base_b +
    (((b_lane_row_base + wk) + (B_col_base[1])*BK) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(rb_0_1_0), "=r"(rb_0_1_1),
          "=r"(rb_1_1_0), "=r"(rb_1_1_1)
        : "r"(lb_0_and_1_1)
    );

    uint32_t lb_0_and_1_2 =
    smem_base_b +
    (((b_lane_row_base + wk) + (B_col_base[2])*BK) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(rb_0_2_0), "=r"(rb_0_2_1),
          "=r"(rb_1_2_0), "=r"(rb_1_2_1)
        : "r"(lb_0_and_1_2)
    );

    uint32_t lb_0_and_1_3 =
    smem_base_b +
    (((b_lane_row_base + wk) + (B_col_base[3])*BK) * sizeof(nv_bfloat16));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(rb_0_3_0), "=r"(rb_0_3_1),
          "=r"(rb_1_3_0), "=r"(rb_1_3_1)
        : "r"(lb_0_and_1_3)
    );

    // lmaoo I dont know what is worse this or caving in and using cutlass

    //stage 0 matmuls 
    //m_acc=0,n_acc=0
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_0_0), "=f"(rc_0_0_1), "=f"(rc_0_0_2), "=f"(rc_0_0_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_0_0), "r"(rb_0_0_1),
          "f"(rc_0_0_0), "f"(rc_0_0_1), "f"(rc_0_0_2), "f"(rc_0_0_3))
    );
    //m_acc=0,n_acc=1
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_1_0), "=f"(rc_0_1_1), "=f"(rc_0_1_2), "=f"(rc_0_1_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_1_0), "r"(rb_0_1_1),
          "f"(rc_0_1_0), "f"(rc_0_1_1), "f"(rc_0_1_2), "f"(rc_0_1_3))
    );
    //m_acc=0,n_acc=2
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_2_0), "=f"(rc_0_2_1), "=f"(rc_0_2_2), "=f"(rc_0_2_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_2_0), "r"(rb_0_2_1),
          "f"(rc_0_2_0), "f"(rc_0_2_1), "f"(rc_0_2_2), "f"(rc_0_2_3))
    );
     //m_acc=0,n_acc=3
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_3_0), "=f"(rc_0_3_1), "=f"(rc_0_3_2), "=f"(rc_0_3_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_3_0), "r"(rb_0_3_1),
          "f"(rc_0_3_0), "f"(rc_0_3_1), "f"(rc_0_3_2), "f"(rc_0_3_3))
    );

    //m_acc=1,n_acc=0
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_1_0_0), "=f"(rc_1_0_1), "=f"(rc_1_0_2), "=f"(rc_1_0_3)
        : "r"(ra_0_1_0), "r"(ra_0_1_1), "r"(ra_0_1_2), "r"(ra_0_0_3),
          "r"(rb_0_0_0), "r"(rb_0_0_1),
          "f"(rc_0_0_0), "f"(rc_0_0_1), "f"(rc_0_0_2), "f"(rc_0_0_3))
    );
    //m_acc=0,n_acc=1
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_1_0), "=f"(rc_0_1_1), "=f"(rc_0_1_2), "=f"(rc_0_1_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_1_0), "r"(rb_0_1_1),
          "f"(rc_0_1_0), "f"(rc_0_1_1), "f"(rc_0_1_2), "f"(rc_0_1_3))
    );
    //m_acc=0,n_acc=2
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_2_0), "=f"(rc_0_2_1), "=f"(rc_0_2_2), "=f"(rc_0_2_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_2_0), "r"(rb_0_2_1),
          "f"(rc_0_2_0), "f"(rc_0_2_1), "f"(rc_0_2_2), "f"(rc_0_2_3))
    );
     //m_acc=0,n_acc=3
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(rc_0_3_0), "=f"(rc_0_3_1), "=f"(rc_0_3_2), "=f"(rc_0_3_3)
        : "r"(ra_0_0_0), "r"(ra_0_0_1), "r"(ra_0_0_2), "r"(ra_0_0_3),
          "r"(rb_0_3_0), "r"(rb_0_3_1),
          "f"(rc_0_3_0), "f"(rc_0_3_1), "f"(rc_0_3_2), "f"(rc_0_3_3))
    );
  }


}

int main()
{
  return 0;
}

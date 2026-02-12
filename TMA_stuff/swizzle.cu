#include "../atoms/all.cuh"
struct SwizzleConfig
{
    using dtype = float;

    static constexpr int M = 16;
    static constexpr int N = 16;

    // Box tile == full matrix for now
    static constexpr int BOX_M = M;
    static constexpr int BOX_N = N;

    // Input swizzle (experiment knob)
    static constexpr CUtensorMapSwizzle IN_SWIZZLE =
        CU_TENSOR_MAP_SWIZZLE_64B;

    // Output always identity
    static constexpr CUtensorMapSwizzle OUT_SWIZZLE =
        CU_TENSOR_MAP_SWIZZLE_NONE;
};

namespace ptx = cuda::ptx; 
using barrier = cuda::barrier<cuda::thread_scope_block>;

template <class cfg>
__global__ void copy(const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap a_out_map)
{
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  
  using T = typename cfg::dtype;
  uint8_t* ptr = smem_raw;
  T* As = alloc<T, 1024>(ptr, cfg::M * cfg::N);

  int l = threadIdx.x; 
  __shared__ barrier bar; 
  barrier::arrival_token token;
  if (l == 0)
    {
      init(&bar, blockDim.x);
    }
    __syncthreads();
  
  int32_t A_coords[2] = {0, 0};

  if(l == 0)
  {
    ptx::cp_async_bulk_tensor(
    ptx::space_shared, ptx::space_global,
      As, &a_map, A_coords, cuda::device::barrier_native_handle(bar));

    token = cuda::device::barrier_arrive_tx(bar, 1, cfg::M*cfg::N*sizeof(T));
    
  }
  else
  {
    token = bar.arrive();
  }
  bar.wait(std::move(token)); 
  asm volatile("fence.proxy.async.shared::cta;");

  if(l == 0)
  {
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_global, cuda::ptx::space_shared,
        &a_out_map, A_coords, As
      );
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  }
  
}

int main()
{
    using Cfg = SwizzleConfig;
    using T   = Cfg::dtype;

    NaiveTensor<T> A({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);
    NaiveTensor<T> Out({Cfg::M, Cfg::N}, Layout::ROW_MAJOR);

    A.allocate();
    Out.allocate();

    A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1);
    Out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);

    A.to_device();
    Out.to_device();

    // ============================================================
    // RAW tensor map parameters (explicit)
    // ============================================================

    // Rank = 2
    constexpr uint32_t rank = 2;

    // Driver expects dims in order: [fastest, slowest]
    // Row-major: fastest = N, slowest = M
    uint64_t global_dims[2] = {
        uint64_t(Cfg::N),   // dim0 (fastest)
        uint64_t(Cfg::M)    // dim1 (slowest)
    };

    // Strides: only stride for dim1 (slowest), in BYTES
    uint64_t global_strides[1] = {
        uint64_t(Cfg::N) * sizeof(T)
    };

    // Box dims also ordered [fastest, slowest]
    uint32_t box_dims[2] = {
        uint32_t(Cfg::BOX_N),
        uint32_t(Cfg::BOX_M)
    };

    // Element strides always {1,1}
    uint32_t elem_strides[2] = {1, 1};

    // ============================================================
    // Build INPUT map (swizzled)
    // ============================================================

    CUtensorMap in_map =
        TmaDescriptor<T>::create_raw(
            A.d_ptr,
            rank,
            global_dims,
            global_strides,
            box_dims,
            elem_strides,
            Cfg::IN_SWIZZLE,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE
        );

    // ============================================================
    // Build OUTPUT map (identity)
    // ============================================================

    CUtensorMap out_map =
        TmaDescriptor<T>::create_raw(
            Out.d_ptr,
            rank,
            global_dims,
            global_strides,
            box_dims,
            elem_strides,
            Cfg::OUT_SWIZZLE,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE
        );

    printf("Raw tensor maps built.\n");

    NaiveLauncher launcher(1,1,32,Cfg::M*Cfg::N*sizeof(T)); 
    launcher.launch(copy<Cfg>,in_map,out_map);

    cudaDeviceSynchronize();

    Out.to_host();
    Out.pretty_print();
}

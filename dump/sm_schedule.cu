#include "../atoms/naive_tensor.cuh"
#include "../atoms/utils.cuh"
#include "../atoms/naive_launcher.cuh"
namespace cg = cooperative_groups;


//#the int SM_id array, is going to be a 2d_array n_clusters_in_grid x n_blocks_in_cluster
//#and will depict the (cluster_id, block_in_cluster_rank) -> sm_id(cluster_id,block_in_cluster_rank)

constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 2048;

constexpr int threads_per_block = 1024; 
constexpr int blocks_per_cluster = 1; 
constexpr int clusters_per_grid = (M*N)/(threads_per_block*blocks_per_cluster);


__global__ void matmul_with_sm_schedule(float*A,float*B,float*C, uint32_t*SM_id)
{
  auto g = cg::this_grid();
  unsigned int cluster_rank = g.cluster_rank(); 
  auto c = cg::this_cluster();
  unsigned int block_in_cluster_rank = c.block_rank();

  int global_thread = g.thread_rank();
  int m = global_thread/N; 
  int n = global_thread%N; 
  float tmp = 0.0;
  for (int k = 0; k < K; k++)
  {
    tmp += A[m*K + k]*B[k*N + n];
  }
  C[m*N + n] = tmp; 
  if (threadIdx.x == 0)
  {
    uint32_t sm_id = cuda::ptx::get_sreg_smid();
    SM_id[cluster_rank*blocks_per_cluster + block_in_cluster_rank] = sm_id;
  }

}


int main()
{
  NaiveTensor<float>A(M*K);
  NaiveTensor<float>B(K*N); 
  NaiveTensor<float>C(M*N);

  A.allocate();
  B.allocate();
  C.allocate(); 
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  A.to_device();
  B.to_device();
  C.to_device();

  NaiveTensor<uint32_t>SM_map(clusters_per_grid*blocks_per_cluster);
  SM_map.allocate(); 
  SM_map.to_device();
  NaiveLauncher launcher(clusters_per_grid, blocks_per_cluster, threads_per_block,0);
  launcher.launch(matmul_with_sm_schedule,A.d_ptr,B.d_ptr,C.d_ptr,SM_map.d_ptr);
  cudaDeviceSynchronize();
  A.to_host();
  B.to_host();
  C.to_host();
  SM_map.to_host();
  SM_map.pretty_print({clusters_per_grid, blocks_per_cluster});
}
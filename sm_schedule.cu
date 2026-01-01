#include "atoms/naive_tensor.cuh"
#include "atoms/utils.cuh"
#include "atoms/naive_launcher.cuh"
namespace cg = cooperative_groups

//#the int SM_id array, is going to be a 2d_array n_clusters_in_grid x n_blocks_in_cluster
//#and will depict the (cluster_id, block_in_cluster_rank) -> sm_id(cluster_id,block_in_cluster_rank)

constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 2048;

constexpr int threads_per_block = 1024; 
constexpr int blocks_per_cluster = 2; 
constexpr int clusters_per_grid = (M*N)/(threads_per_block*blocks_per_cluster);


__global__ void matmul_with_sm_schedule(float*A,float*B,float*C, int*SM_id)
{
  auto g = cg::this_grid();
  unsigned int cluster_rank = g.cluster_rank(); 
  auto c = cg::this_cluster();
  unsigned int block_in_cluster_rank = c.block_rank();
  auto b = cg::this_block();
  unsigned int thread_in_block_rank = b.thread_rank();
  int global_thread = g.thread_rank();
  int m = global_thread/N; 
  int n = global_thread%N; 
  float tmp = 0.0;
  for (int k = 0; k < K; k++)
  {
    tmp += A[m*K + k]*B[k*N + n];
  }
  C[m*N + n] = temp; 
  if (threadIdx.x == 0)
  {

  }

}


int main()
{

}
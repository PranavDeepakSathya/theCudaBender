#include "../atoms/all.cuh"
constexpr int N = 1024*1024; 
constexpr int block_size = 1024;
constexpr int grid_size = N/block_size;

__global__ void naive_reduce(float*A, float*B)
{
  int t = threadIdx.x; 
  int b = blockIdx.x; 
  int global_off = (b*block_size) + t 
  __shared__ As[block_size]; 
  As[t] = A[global_off]; 
   
}


int main()
{
  NaiveTensor<float>A({N}, Layout::ROW_MAJOR); 
  NaiveTensor<float>A_out({1}, Layout::ROW_MAJOR); 
  printf("grid_size %d \n",grid_size);

}
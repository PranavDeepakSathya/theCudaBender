#include "../atoms/all.cuh"
constexpr int N = 256; 
constexpr int block_size = 64;
constexpr int grid_size = N/block_size;

__global__ void naive_reduce(float*A, float*B)
{
  int t = threadIdx.x; 
  int b = blockIdx.x; 
  int b_dim = blockDim.x;
  int global_off = (b*block_size) + t;
  __shared__ float As[block_size]; 
  As[t] = A[global_off];
  __syncthreads();

  for (int s = 1; s < block_size; s*=2)
  {
    if ((t % (2*s) == 0) && t + s < block_size)
    {
      As[t] = As[t] + As[t + s];
    }
    __syncthreads(); 
  }
  
  if (t == 0)
  {
    B[b + t] = As[t];
  }
}

__global__ void second_naive_reduce(float*A, float*B)
{
  int t = threadIdx.x; 
  int b = blockIdx.x; 
  int b_dim = blockDim.x;
  int global_off = (b*grid_size) + t;
  __shared__ float As[grid_size]; 
  As[t] = A[global_off];
  __syncthreads();

  for (int s = 1; s < grid_size; s*=2)
  {
    if ((t % (2*s) == 0) && t + s < grid_size)
    {
      As[t] = As[t] + As[t + s];
    }
    __syncthreads(); 
  }
  
  if (t == 0)
  {
    B[b + t] = As[t];
  }
}

int main()
{
  NaiveTensor<float>A({N}, Layout::ROW_MAJOR); 
  NaiveTensor<float>A_mid({grid_size}, Layout::ROW_MAJOR);
  NaiveTensor<float>A_out({1}, Layout::ROW_MAJOR); 
  printf("grid_size %d \n",grid_size);

  A.allocate(); 
  A_mid.allocate(); 
  A_out.allocate(); 

  A.init_pattern(MODE_ARANGE,DIST_FLOAT_NEG1_1);
  A_mid.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 
  A_out.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 

  A.to_device(); 
  A_mid.to_device();
  A_out.to_device(); 

  naive_reduce<<<grid_size,block_size>>>(A.d_ptr, A_mid.d_ptr); 
  cudaDeviceSynchronize(); 
  second_naive_reduce<<<1,grid_size>>>(A_mid.d_ptr, A_out.d_ptr);

  A_out.to_host();

  A_out.pretty_print();
  printf("ans : %d", (N*(N-1)/2));

}
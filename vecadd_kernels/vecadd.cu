#include "../atoms/all.cuh"

constexpr int N = 4096; 
constexpr int block_size = 256; 
constexpr int grid_size = N/block_size;

__global__ void vec_add (nv_bfloat16*A, nv_bfloat16 *B, nv_bfloat16*C)
{
  int global_tid = (blockIdx.x*block_size) + threadIdx.x; 

  C[global_tid] = A[global_tid] + B[global_tid];
}

int main()
{
  NaiveTensor<nv_bfloat16>A({N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>B({N}, Layout::ROW_MAJOR); 
  NaiveTensor<nv_bfloat16>C({N}, Layout::ROW_MAJOR);

  A.allocate(); 
  B.allocate(); 
  C.allocate(); 

  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1); 
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1); 

  A.to_device(); 
  B.to_device(); 
  C.to_device(); 

  vec_add<<<grid_size, block_size>>>(A.d_ptr, B.d_ptr, C.d_ptr); 
  cudaDeviceSynchronize(); 

  C.to_host(); 

  printf("\n===========A============\n"); 
  A.pretty_print();
  printf("\n============B============\n"); 
  B.pretty_print(); 
  printf("\n============C============\n"); 
  C.pretty_print();


}
#include "../atoms/utils.cuh"
#include "../atoms/naive_tensor.cuh"

__global__ void vec_add (nv_bfloat16* A, nv_bfloat16 *B, nv_bfloat16 *C)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x; 
  C[x] = A[x] + B[x];
  
}

int main()
{
  NaiveTensor<nv_bfloat16> A(1024); 
  NaiveTensor<nv_bfloat16> B(1024);
  NaiveTensor<nv_bfloat16> C(1024);

  A.allocate();
  B.allocate();
  C.allocate();
  A.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  B.init_pattern(MODE_RAND, DIST_FLOAT_NEG1_1);
  C.init_pattern(MODE_ZEROS, DIST_FLOAT_NEG1_1);
  A.to_device();
  B.to_device();
  C.to_device();
  vec_add<<<1024/256,256>>>(A.d_ptr, B.d_ptr, C.d_ptr);
  cudaDeviceSynchronize();
  C.to_host();
  for (int i = 0; i < 10; i++)
  {
    
    C.print_sample(i,"C");
    A.print_sample(i, "A");
    B.print_sample(i, "B");
    printf("\n");
    printf("error at %d, %f", i,(float)(C.h_ptr[i]-(A.h_ptr[i]+B.h_ptr[i])));
    printf("\n");
    
  }


}
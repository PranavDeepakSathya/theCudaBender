#include "atoms/utils.cuh"
#include "atoms/naive_tensor.cuh"



int main()
{
  NaiveTensor<nv_bfloat16> A(1024); 
  NaiveTensor<nv_bfloat16> B(1024);
  A.allocate();
  B.allocate();

}
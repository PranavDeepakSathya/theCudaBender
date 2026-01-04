#include "../atoms/utils.cuh"
#include "../atoms/naive_tensor.cuh"

__global__ void test_kernel(NaiveTensor<nv_bfloat16>::DeviceView A)
{
  int t = threadIdx.x; 
  int t0 = t % 3; 
  int t1 = ((int)(t/3) % 4); 
  nv_bfloat16 v = A.get(t0,t1);
  printf("%d,-> (%d, %d) : %f \n", t, t0,t1, (float)v);

}

int main()
{
  NaiveTensor<nv_bfloat16>A({3,4}, Layout::COL_MAJOR); 
  A.allocate(); 
  A.init_pattern(MODE_ARANGE, DIST_INT_0_100); 
  nv_bfloat16 val = A.get_host(1,0); 
  printf("%f\n",(float) val);
  auto A_view = A.get_device_view();
  A.to_device();
  test_kernel<<<1,12>>>(A_view); 
  cudaDeviceSynchronize();

}

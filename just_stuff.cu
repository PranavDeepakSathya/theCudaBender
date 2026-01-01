
#include<stdio.h> 
#include<cuda.h> 
#include<cuda_runtime.h> 
#include<cuda_fp16.h> 
#include<cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
enum InitMode
{
  MODE_ZEROS, 
  MODE_RAND, 
  MODE_ARANGE,
};

template <typename T> 
void init_array_host_to_device(T*& host_null_ptr, T*& device_null_ptr, int N_elems, InitMode mode)
{
  CHECK_CUDA(cudaHostAlloc(&host_null_ptr, N_elems*sizeof(T), cudaHostAllocDefault));
  switch (mode)
  {
    case MODE_ZEROS:
      break;
    case MODE_ARANGE: 
      for (int i = 0; i < N_elems; i++) host_null_ptr[i] = static_cast<T>(i);
      break;
    case MODE_RAND: 
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(int i = 0; i < N_elems; ++i) host_null_ptr[i] = static_cast<T>(dist(gen));
      } else {
        std::uniform_int_distribution<int> dist(0, 100);
        for(int i = 0; i < N_elems; ++i) host_null_ptr[i] = static_cast<T>(dist(gen));
      }
      break;
    }

  }

}

int main()
{
  nv_bFloat16* A_h, *A_d; 
  A_h = nullptr; 
  A_d = nullptr;
  init_array_host_to_device(A_h, A_d, 10, MODE_ARANGE);
  printf("\n %f \n",A_h[3]);
  
  return 0;
}
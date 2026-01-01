#include<stdio.h> 
#include<cuda.h> 
#include<cuda_runtime.h> 
#include<cuda_fp16.h> 
#include<cuda_bf16.h>
#include <iostream>
#include <random>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

enum InitMode {
  MODE_ZEROS, 
  MODE_ARANGE,
  MODE_RAND
};

// New Enum to control the random generation logic explicitly
enum RandomDist {
    DIST_FLOAT_NEG1_1, // Generate floats [-1.0, 1.0]
    DIST_INT_0_100     // Generate ints [0, 100]
};

template <typename T> 
void init_array_host_to_device(T*& h_ptr, T*& d_ptr, int N_elems, 
                               InitMode mode, 
                               RandomDist rand_dist = DIST_FLOAT_NEG1_1) // Default to float behavior
{
  size_t bytes = N_elems * sizeof(T);
  CHECK_CUDA(cudaHostAlloc(&h_ptr, bytes, cudaHostAllocDefault));

  switch (mode)
  {
    case MODE_ZEROS:
      memset(h_ptr, 0, bytes);
      break;

    case MODE_ARANGE: 
      for (int i = 0; i < N_elems; i++) h_ptr[i] = static_cast<T>(i);
      break;

    case MODE_RAND: 
    {
      std::random_device rd;
      std::mt19937 gen(rd());

      // We switch based on the EXPLICIT request, not the type T.
      // This allows you to fill "int8" with floats (that get truncated),
      // or "fp8" with integer values if you want.
      if (rand_dist == DIST_FLOAT_NEG1_1) {
         std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
         for(int i = 0; i < N_elems; ++i) 
             h_ptr[i] = static_cast<T>(dist(gen)); // Cast handles Value Conversion
      } else {
         std::uniform_int_distribution<int> dist(0, 100);
         for(int i = 0; i < N_elems; ++i) 
             h_ptr[i] = static_cast<T>(dist(gen));
      }
      break;
    }
  }

  CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
  CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
}

__global__ void add_one (half* A)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  A[x] = A[x] + A[x];
}

int main()
{
  half *A_h = nullptr, *A_d = nullptr; 
  
  // Example 1: Treat BF16 as Float (standard)
  init_array_host_to_device(A_h, A_d, 1024, MODE_RAND, DIST_FLOAT_NEG1_1);
  printf("BF16 Float Mode: %f\n", (float)A_h[2]);

  add_one<<<1,1024>>>(A_d);
  cudaDeviceSynchronize();
  cudaMemcpy(A_h, A_d, sizeof(nv_bfloat16)*1024, cudaMemcpyDeviceToHost);
  printf("BF16 Float Mode: %f\n", (float)A_h[2]);
  // Cleanup to reuse pointers for demo
  cudaFreeHost(A_h); cudaFree(A_d);


  

  return 0;
}
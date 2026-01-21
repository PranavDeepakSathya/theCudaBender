#include "atoms/all.cuh"

constexpr int ld_n = 8;
constexpr int ld_m = 8;
constexpr int x = 4;
constexpr int m = ld_m*ld_n*x; 
constexpr int loads_per_lane = 2*x; 
constexpr int loads_per_iter = 32;

__global__ void reg_layout (NaiveTensor<nv_bfloat16>::DeviceView A)
{
  int l = threadIdx.x % 32; 
  __shared__ nv_bfloat16 As[m]; 
  
  for (int i = 0; i < loads_per_lane; i++)
  {
    As[l +(i*loads_per_iter)] = A.get(l +(i*loads_per_iter));
    printf("%f \n", (float)As[l +(i*loads_per_iter)]);
  }

  int lane_row = (l%(x*ld_n))*ld_m;
  uint32_t smem_base_A = static_cast<uint32_t>(__cvta_generic_to_shared(As));
  uint32_t ld_addr = smem_base_A + (lane_row * sizeof(nv_bfloat16));

  nv_bfloat162 rA[1];
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rA);

    asm volatile(
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
    : "=r"(reg_ptr[0]) 
    : "r"(ld_addr)
  );

  uint32_t packed = reg_ptr[0];

  uint16_t lo = packed & 0xFFFF;
  uint16_t hi = packed >> 16;

  nv_bfloat16 a = *reinterpret_cast<nv_bfloat16*>(&lo);
  nv_bfloat16 b = *reinterpret_cast<nv_bfloat16*>(&hi);

  printf("lane %d | rA = (%f, %f)\n",
        l, (float)a, (float)b);


}

int main()
{
  NaiveTensor<nv_bfloat16>A({m}, Layout::ROW_MAJOR); 
  A.init_pattern(MODE_ARANGE, DIST_FLOAT_NEG1_1); 
  A.to_device(); 
  auto A_view = A.get_device_view(); 
  reg_layout<<<1,32>>>(A_view); 
  cudaDeviceSynchronize(); 
  
  

}
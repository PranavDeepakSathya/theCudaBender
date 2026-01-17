#include "../atoms/all.cuh"

constexpr int n_blocks = 1; 
constexpr int n_tpb = 32; 

__device__ __forceinline__
nv_bfloat16 bf(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ nv_bfloat162 pack(nv_bfloat16 x, nv_bfloat16 y) {
    return make_bfloat162(x, y);
}

__device__ __forceinline__
void mma_m16n8k16_f32_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__
void unpack_bf162(nv_bfloat162 v, float &x, float &y)
{
    // reinterpret bits
    uint32_t bits = reinterpret_cast<uint32_t&>(v);

    // low 16 bits, high 16 bits
    uint16_t lo = bits & 0xFFFF;
    uint16_t hi = bits >> 16;

    nv_bfloat16 b0 = reinterpret_cast<nv_bfloat16&>(lo);
    nv_bfloat16 b1 = reinterpret_cast<nv_bfloat16&>(hi);

    x = __bfloat162float(b0);
    y = __bfloat162float(b1);
}

__device__ __forceinline__
void pretty_print_lane(
    int lane,
    nv_bfloat162 A0, nv_bfloat162 A1, nv_bfloat162 A2, nv_bfloat162 A3,
    nv_bfloat162 B0, nv_bfloat162 B1,
    float c0, float c1, float c2, float c3)
{
    float a00,a01,a10,a11,a20,a21,a30,a31;
    float b00,b01,b10,b11;

    unpack_bf162(A0, a00, a01);
    unpack_bf162(A1, a10, a11);
    unpack_bf162(A2, a20, a21);
    unpack_bf162(A3, a30, a31);

    unpack_bf162(B0, b00, b01);
    unpack_bf162(B1, b10, b11);

    printf(
        "lane %2d | "
        "A: [%4.4f %4.4f | %4.4f %4.4f | %4.4f %4.4f | %4.4f %4.4f] "
        "B: [%4.4f %4.4f | %4.4f %4.4f] "
        "C: [%8.4f %8.4f %8.4f %8.4f]\n",
        lane,
        a00,a01,a10,a11,a20,a21,a30,a31,
        b00,b01,b10,b11,
        c0,c1,c2,c3
    );
}

__global__ void probe_kernel(int la,int ra,int pa,int lb,int rb,int pb)
{
  int l = threadIdx.x & 31;

    // ------------------
    // initialize A = 0
    // ------------------
    float a[4][2] = {0};

    if (l == la) {
        a[ra][pa] = 1.0f;
    }

    nv_bfloat162 A0 = make_bfloat162(bf(a[0][0]), bf(a[0][1]));
    nv_bfloat162 A1 = make_bfloat162(bf(a[1][0]), bf(a[1][1]));
    nv_bfloat162 A2 = make_bfloat162(bf(a[2][0]), bf(a[2][1]));
    nv_bfloat162 A3 = make_bfloat162(bf(a[3][0]), bf(a[3][1]));

    // ------------------
    // initialize B = 0
    // ------------------
    float b[2][2] = {0};

    if (l == lb) {
        b[rb][pb] = 1.0f;
    }

    nv_bfloat162 B0 = make_bfloat162(bf(b[0][0]), bf(b[0][1]));
    nv_bfloat162 B1 = make_bfloat162(bf(b[1][0]), bf(b[1][1]));

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    mma_m16n8k16_f32_bf16(
        c0,c1,c2,c3,
        reinterpret_cast<uint32_t&>(A0),
        reinterpret_cast<uint32_t&>(A1),
        reinterpret_cast<uint32_t&>(A2),
        reinterpret_cast<uint32_t&>(A3),
        reinterpret_cast<uint32_t&>(B0),
        reinterpret_cast<uint32_t&>(B1),
        0,0,0,0
    );


    if (fabsf(c0) > 0.5f)
        printf("(%d,%d,%d)x(%d,%d,%d) -> (%d,0)\n",
              la, ra, pa,
              lb, rb, pb,
              l);

    if (fabsf(c1) > 0.5f)
        printf("(%d,%d,%d)x(%d,%d,%d) -> (%d,1)\n",
              la, ra, pa,
              lb, rb, pb,
              l);

    if (fabsf(c2) > 0.5f)
        printf("(%d,%d,%d)x(%d,%d,%d) -> (%d,2)\n",
              la, ra, pa,
              lb, rb, pb,
              l);

    if (fabsf(c3) > 0.5f)
        printf("(%d,%d,%d)x(%d,%d,%d) -> (%d,3)\n",
              la, ra, pa,
              lb, rb, pb,
              l);
}


int main()
{
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1 << 26); // big printf buffer

    for (int la = 0; la < 32; ++la)
    for (int ra = 0; ra < 4;  ++ra)
    for (int pa = 0; pa < 2;  ++pa)
    {
        for (int lb = 0; lb < 32; ++lb)
        for (int rb = 0; rb < 2;  ++rb)
        for (int pb = 0; pb < 2;  ++pb)
        {
            probe_kernel<<<1, 32>>>(la, ra, pa, lb, rb, pb);
            cudaDeviceSynchronize();
        }
    }

    cudaDeviceReset();
    return 0;
}
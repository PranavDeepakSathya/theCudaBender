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

__global__ void test_kernel()
{
    int l = threadIdx.x % 32; 
    float a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h, b0l, b0h, b1l, b1h;
    float lane = float(l);

    a0l = lane * 0.25f + 0.01f;
    a0h = lane * 0.25f + 0.02f;

    a1l = lane * 0.25f + 0.03f;
    a1h = lane * 0.25f + 0.04f;

    a2l = lane * 0.25f + 0.05f;
    a2h = lane * 0.25f + 0.06f;

    a3l = lane * 0.25f + 0.07f;
    a3h = lane * 0.08f;

    b0l = lane * 0.25f + 0.11f;
    b0h = lane * 0.25f + 0.12f;

    b1l = lane * 0.25f + 0.13f;
    b1h = lane * 0.25f + 0.14f;



    nv_bfloat162 A0 = pack(bf(a0l), bf(a0h));
    nv_bfloat162 A1 = pack(bf(a1l), bf(a1h));
    nv_bfloat162 A2 = pack(bf(a2l), bf(a2h));
    nv_bfloat162 A3 = pack(bf(a3l), bf(a3h));

    uint32_t a0 = reinterpret_cast<uint32_t&>(A0);
    uint32_t a1 = reinterpret_cast<uint32_t&>(A1);
    uint32_t a2 = reinterpret_cast<uint32_t&>(A2);
    uint32_t a3 = reinterpret_cast<uint32_t&>(A3);
    nv_bfloat162 B0 = pack(bf(b0l), bf(b0h));
    nv_bfloat162 B1 = pack(bf(b1l), bf(b1h));

    uint32_t b0 = reinterpret_cast<uint32_t&>(B0);
    uint32_t b1 = reinterpret_cast<uint32_t&>(B1);
    float c0 = 0.f;
    float c1 = 0.f;
    float c2 = 0.f;
    float c3 = 0.f;
    mma_m16n8k16_f32_bf16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,c0,c1,c2,c3);
    pretty_print_lane(
    l,
    A0,A1,A2,A3,
    B0,B1,
    c0,c1,c2,c3
);

}

int main()
{
    test_kernel<<<n_blocks, n_tpb>>>(); 
    cudaDeviceSynchronize();

}
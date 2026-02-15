
import torch
import time
from torch.utils.cpp_extension import load

ext = load(
    name="folder_matmul",
    sources=["export.cu"],
    extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

torch.manual_seed(0)
device = "cuda"

M, N, K = ext.shape()
print(f"\nUsing shape: M={M}, N={N}, K={K}")

A = torch.randn((M, K), device=device, dtype=torch.bfloat16).contiguous()

B_rm = torch.randn((N, K), device=device, dtype=torch.bfloat16)
B = B_rm.t() 

print("\n=== Correctness ===")

torch.cuda.synchronize()
C = ext.gemm(A, B)
torch.cuda.synchronize()

C_ref = A.float() @ B.float()

diff = (C - C_ref).abs().flatten()

qs = torch.tensor([0.50, 0.90, 0.99, 0.999, 1.00], device="cuda")
vals = torch.quantile(diff, qs).cpu().tolist()

print("\n=== Error Percentiles (abs) ===")
for q, v in zip(qs.cpu().tolist(), vals):
    print(f"{q*100:6.2f}% : {v:.6e}")


print("\n=== Benchmark ===")

warmup = 20
iters = 200

for _ in range(warmup):
    ext.gemm(A, B)

torch.cuda.synchronize()
t0 = time.time()

for _ in range(iters):
    ext.gemm(A, B)

torch.cuda.synchronize()
t1 = time.time()

avg_ms = (t1 - t0) * 1e3 / iters

flops = 2.0 * M * N * K
tflops = flops / ((avg_ms * 1e-3) * 1e12)

print(f"Avg time: {avg_ms:.4f} ms")
print(f"TFLOP/s : {tflops:.2f}")

print("\n=== Tensor Debug ===")
print("A stride:", A.stride(), "contig:", A.is_contiguous())
print("B stride:", B.stride(), "contig:", B.is_contiguous())
print("C has NaN:", torch.isnan(C).any().item())

print("\nDone.")

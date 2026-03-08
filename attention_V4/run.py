import torch
import time
from torch.utils.cpp_extension import load
import triton.testing as testing
import math

ext = load(
    name="attention_warp",
    sources=["export.cu"],
    extra_cuda_cflags=[
        "-arch=sm_120",
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

torch.manual_seed(0)
device = "cuda"

BH, LQ, D, LK = ext.shape()
print(f"BH={BH}, LQ={LQ}, D={D}, LK={LK}")

# -----------------------------
# Inputs
# -----------------------------

Q = torch.randn((BH, LQ, D), device="cuda", dtype=torch.bfloat16).contiguous()
K = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)
V = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)

assert V.stride(1) == 1
assert K.stride(1) == 1

# -----------------------------
# Kernel
# -----------------------------

O = ext.attention(Q, K, V)

# -----------------------------
# Reference
# -----------------------------

S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / math.sqrt(D)
P = torch.softmax(S, dim=-1)
O_ref = torch.matmul(P, V.float())

diff = (O - O_ref).abs()

print("Max abs error:", diff.max().item())
print("Mean abs error:", diff.mean().item())

# how many elements exceed thresholds
for t in [1e-3, 1e-2, 5e-2, 1e-1]:
    count = (diff > t).sum().item()
    print(f">{t}: {count}")
    
# reshape helpers
BH_, LQ_, D_ = O.shape

# thresholds
bands = [
    ("1e-3", diff > 1e-3),
    ("1e-2", diff > 1e-2),
    ("5e-2", diff > 5e-2),
]

for name, mask in bands:
    idx = torch.nonzero(mask)

    print(f"\n==== elements > {name} ====")
    print("count:", idx.shape[0])

    if idx.shape[0] == 0:
        continue

    # show a few coordinates
    print("sample coordinates (BH, LQ, D):")
    print(idx[:10])

    # compute rough ranges
    bh_min, bh_max = idx[:,0].min().item(), idx[:,0].max().item()
    lq_min, lq_max = idx[:,1].min().item(), idx[:,1].max().item()
    d_min,  d_max  = idx[:,2].min().item(), idx[:,2].max().item()

    print("BH range:", bh_min, bh_max)
    print("LQ range:", lq_min, lq_max)
    print("D range:", d_min, d_max)
    
  
for seed in range(10):
    torch.manual_seed(seed)

    Q = torch.randn((BH, LQ, D), device="cuda", dtype=torch.bfloat16).contiguous()
    K = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)
    V = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)

    O = ext.attention(Q, K, V)

    S = torch.matmul(Q.float(), K.float().transpose(-1,-2)) / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V.float())

    diff = (O - O_ref).abs()

    print("Max abs error:", diff.max().item())
    print("Mean abs error:", diff.mean().item())
    print (diff.flatten()[9000000:9000100])




def run():
    ext.attention(Q, K, V)

ms = testing.do_bench(run, return_mode="median")  # median

flops = 4 * BH * LQ * LK * D
tflops = flops / (ms * 1e-3) / 1e12

print("Effective TFLOP/s:", tflops)


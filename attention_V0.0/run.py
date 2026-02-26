import torch
import time
from torch.utils.cpp_extension import load

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

LQ, D, LK = ext.shape()
print(f"LQ={LQ}, D={D}, LK={LK}")

# Q: row-major (LQ, D)
Q = torch.randn((LQ, D), device=device, dtype=torch.bfloat16).contiguous()

# K: row-major (LK, D)
K = torch.randn((LK, D), device=device, dtype=torch.bfloat16).contiguous()

# V: column-major (LK, D)
V_rm = torch.randn((D, LK), device=device, dtype=torch.bfloat16)
V = V_rm.t()  # now (LK, D) with stride(0) == 1
# correctness
O = ext.attention(Q, K, V)

S = (Q.float() @ K.float())
P = torch.softmax(S, dim=1)
O_ref = P @ V.float()

diff = (O - O_ref).abs()

print("Max abs error:", diff.max().item())

# benchmark
iters = 2000
torch.cuda.synchronize()
t0 = time.time()

for _ in range(iters):
    ext.attention(Q, K, V)

torch.cuda.synchronize()
t1 = time.time()

print("Avg us:", (t1 - t0)*1e6/iters)
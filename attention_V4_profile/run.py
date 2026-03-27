import torch
import math
import sys
from pathlib import Path
from torch.utils.cpp_extension import load
import triton.testing as testing

# add project root so we can import profiler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profiler.trace import export_trace

# =========================
# BUILD
# =========================

ext = load(
    name="attention_V4_profile",
    sources=["export.cu"],
    extra_cuda_cflags=[
        "-arch=sm_120",
        "-O3",
        "--use_fast_math"
    ],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

torch.manual_seed(0)
device = "cuda"

BH, LQ, D, LK = ext.shape()
WARPS_PER_BLOCK = ext.num_warps()
print(f"BH={BH}, LQ={LQ}, D={D}, LK={LK}, warps_per_block={WARPS_PER_BLOCK}")

# =========================
# INPUTS
# =========================

Q = torch.randn((BH, LQ, D), device=device, dtype=torch.bfloat16).contiguous()
K = torch.randn((BH, LK, D), device=device, dtype=torch.bfloat16).contiguous()
V = torch.randn((BH, D, LK), device=device, dtype=torch.bfloat16).transpose(1,2)
assert V.stride(1) == 1

# =========================
# CORRECTNESS CHECK
# =========================

O, prof_tensor = ext.attention(Q, K, V)

S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / math.sqrt(D)
P = torch.softmax(S, dim=-1)
O_ref = torch.matmul(P, V.float())

diff = (O - O_ref).abs()
print(f"Max abs error: {diff.max().item():.6f}")
print(f"Mean abs error: {diff.mean().item():.6f}")

# =========================
# EXPORT TRACE
# =========================

TAGS = [
    "Q_gmem_to_smem",
    "Q_gmem_to_rmem",
    "KV_gmem_to_smem",
    "K_smem_to_rmem",
    "QK_mma",
    "softmax_and_update",
    "scale",
    "V_smem_to_rmem",
    "PV_mma",
    "store",
]

export_trace(
    prof_tensor,
    tags=TAGS,
    streams_per_block=WARPS_PER_BLOCK,
    granularity="warp",
    output="trace.json",
)

# =========================
# BENCH
# =========================

def run():
    ext.attention(Q, K, V)

ms = testing.do_bench(run, return_mode="median")

flops = 4 * BH * LQ * LK * D
tflops = flops / (ms * 1e-3) / 1e12

print(f"\n{ms:.3f} ms  |  {tflops:.2f} TFLOP/s")

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
        "-I/root/intra-kernel-profiler/include",
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

Q = torch.randn((BH, LQ, D), device=device, dtype=torch.bfloat16).contiguous()
K = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)

# column-major V
V = torch.randn((BH, D, LK), device=device, dtype=torch.bfloat16).transpose(1,2)

assert V.stride(1) == 1

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

for seed in range(10):
    torch.manual_seed(seed)

    Q = torch.randn((BH, LQ, D), device="cuda", dtype=torch.bfloat16).contiguous()
    K = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)
    V = torch.randn((BH, D, LK), device="cuda", dtype=torch.bfloat16).transpose(1,2)

    O = ext.attention(Q, K, V)

    S = torch.matmul(Q.float(), K.float().transpose(-1,-2)) / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V.float())

    print(seed, (O - O_ref).abs().max().item())



# --------------------------------
# IKP Trace: run once and dump
# --------------------------------
print("\n--- IKP Trace Profiling ---")
_ = ext.attention(Q, K, V)  # profiled run
ext.dump_trace("flash_attn_v3_trace.json")
print("Trace written to flash_attn_v3_trace.json")
print("Summary written to flash_attn_v3_trace_summary.json")
print("View trace: open https://ui.perfetto.dev and load flash_attn_v3_trace.json")

# --------------------------------
# Benchmark (trace overhead included but negligible)
# --------------------------------

def run():
    ext.attention(Q, K, V)

ms = testing.do_bench(run, return_mode="median")  # median

flops = 4 * BH * LQ * LK * D
tflops = flops / (ms * 1e-3) / 1e12

print("\nEffective TFLOP/s:", tflops)


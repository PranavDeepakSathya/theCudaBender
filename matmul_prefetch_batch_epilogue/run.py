import torch
import time
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import triton
import triton.testing


# =========================
# GPU + CONFIG
# =========================

GPU_NAME = torch.cuda.get_device_name()
print("\n=== GPU Detected ===")
print(GPU_NAME)

if "5090" in GPU_NAME:
    cfg_file = "5090_cfg.json"
elif "6000" in GPU_NAME:
    cfg_file = "6000_cfg.json"
else:
    raise RuntimeError(f"No config file mapped for GPU: {GPU_NAME}")

print("\nUsing config:", cfg_file)

cfg_path = Path(__file__).parent / cfg_file
with open(cfg_path) as f:
    cfg = json.load(f)

print("\n=== Loaded Config ===")
for k, v in cfg.items():
    print(f"{k} = {v}")

macro_flags = [f"-D{k}={v}" for k, v in cfg.items()]


# =========================
# BUILD EXTENSION
# =========================

ext = load(
    name="matmul_V0",
    sources=["export.cu"],
    extra_cuda_cflags=[
        "-arch=sm_120",
        "-O3",
        "--use_fast_math",
        *macro_flags,
    ],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)


# =========================
# TENSOR SETUP
# =========================

torch.manual_seed(0)
device = "cuda"

L,M, N, K = ext.shape()
print(f"\nUsing shape: L={L}, M={M}, N={N}, K={K}")

A    = torch.randn((L, M, K), device=device, dtype=torch.bfloat16).contiguous()
B_rm = torch.randn((L, N, K), device=device, dtype=torch.bfloat16).contiguous()
B    = torch.transpose(B_rm, -2, -1)
bias = torch.randn((N,), device=device, dtype=torch.float32)


# =========================
# CORRECTNESS
# =========================

print("\n=== Correctness ===")

torch.cuda.synchronize()
C = ext.gemm(A, B, bias)
torch.cuda.synchronize()

C_ref = torch.nn.functional.silu(A.float() @ B.float() + bias)

diff = (C.float() - C_ref).abs().flatten()

# ---- Stable stats (no full sort) ----

numel = diff.numel()
sample_size = min(L*8192*8192, numel)

if sample_size < numel:
    idx = torch.randint(0, numel, (sample_size,), device=device)
    diff_sample = diff[idx]
else:
    diff_sample = diff

max_err = diff_sample.max()
mean_err = diff_sample.mean()
p99 = diff_sample.kthvalue(int(0.99 * diff_sample.numel()))[0]

print("\n=== Error Stats (abs, sampled if large) ===")
print(f"Max  : {max_err.item():.6e}")
print(f"Mean : {mean_err.item():.6e}")
print(f"P99  : {p99.item():.6e}")


# =========================
# BENCHMARK
# =========================

print("\n=== Benchmark ===")

warmup = 20
iters = 2000

for _ in range(warmup):
    ext.gemm(A, B, bias)

torch.cuda.synchronize()
t0 = time.time()

for _ in range(iters):
    ext.gemm(A, B, bias)

torch.cuda.synchronize()
t1 = time.time()

avg_ms = (t1 - t0) * 1e3 / iters

flops = 2.0 * L * M * N * K + L * M * N
tflops = flops / ((avg_ms * 1e-3) * 1e12)

print(f"Avg time: {avg_ms:.4f} ms")
print(f"TFLOP/s : {tflops:.2f}")


# =========================
# DEBUG INFO
# =========================

print("\n=== Tensor Debug ===")
print("A stride:", A.stride(), "contig:", A.is_contiguous())
print("B stride:", B.stride(), "contig:", B.is_contiguous())
print("C has NaN:", torch.isnan(C).any().item())

print("\nDone.")


# =========================
# TORCH BASELINE
# =========================

def bench_torch_baseline(L, M, N, K, device="cuda"):
    A_t    = torch.randn((L, M, K), device=device, dtype=torch.bfloat16).contiguous()
    B_t    = torch.randn((L, K, N), device=device, dtype=torch.bfloat16).contiguous()
    bias_t = torch.randn((N,),      device=device, dtype=torch.float32)

    def eager():
        return torch.nn.functional.silu(torch.bmm(A_t, B_t).float() + bias_t)

    compiled = torch.compile(eager)
    compiled()  # warm up compile

    ms_eager    = triton.testing.do_bench(eager,    return_mode="median")
    ms_compiled = triton.testing.do_bench(compiled, return_mode="median")

    flops = 2 * L * M * N * K + L * M * N
    def tflops(ms): return flops / (ms * 1e-3) / 1e12

    print(f"\n=== torch baseline ===")
    print(f"  eager    : {ms_eager:.4f} ms  {tflops(ms_eager):.2f} TFLOP/s")
    print(f"  compiled : {ms_compiled:.4f} ms  {tflops(ms_compiled):.2f} TFLOP/s")

bench_torch_baseline(L, M, N, K)
print("\n=== Benchmark (Triton do_bench) ===")

def run():
    ext.gemm(A, B, bias)

ms = triton.testing.do_bench(run, return_mode="median")

flops = (2.0 * L * M * N * K) + (4.0 * L * M * N)
tflops = flops / (ms * 1e-3) / 1e12

print(f"median time: {ms:.4f} ms")
print(f"TFLOP/s : {tflops:.2f}")


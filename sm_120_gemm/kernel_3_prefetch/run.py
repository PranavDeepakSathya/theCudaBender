# k0/run.py
import torch
import time
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import triton.testing

GPU_NAME = torch.cuda.get_device_name()
print(f"\n=== GPU: {GPU_NAME} ===")

# ── config: load from results json if exists, else use defaults ───────────────
RESULTS = Path(__file__).parent / "autotune_results.json"

if RESULTS.exists():
    with open(RESULTS) as f:
        best = max(json.load(f), key=lambda r: r["tflops"])
    cfg = best["cfg"]
    print(f"loaded best config from tune: {best['tflops']:.2f} TFLOP/s")
else:
    cfg = {
        "ACC_PER_WARP_M":    2,
        "ACC_PER_WARP_N":    2,
        "WARPS_PER_BLOCK_M": 2,
        "WARPS_PER_BLOCK_N": 4,
        "BLOCK_K":           64,
        "GROUP_M":            4,
        "GROUP_N":            4,
        "BK_STAGES":          2,
        "WK_STAGES":          2
    }
    print("no autotune results found, using defaults")

print("\n=== Config ===")
for k, v in cfg.items():
    print(f"  {k} = {v}")

# ── compile ───────────────────────────────────────────────────────────────────
SRC = str(Path(__file__).parent / "export.cu")
flags = [f"-D{k}={v}" for k, v in cfg.items()]

ext = load(
    name="k0_gemm",
    sources=[SRC],
    extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math", *flags],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

# ── tensors ───────────────────────────────────────────────────────────────────
torch.manual_seed(0)
M, N, K = ext.shape()
print(f"\nshape: M={M} N={N} K={K}")

A    = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).contiguous()
B_rm = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).contiguous()
B    = B_rm.t()

# ── correctness ───────────────────────────────────────────────────────────────
print("\n=== Correctness ===")
torch.cuda.synchronize()
C = ext.gemm(A, B)
torch.cuda.synchronize()

C_ref  = A.float() @ B.float()
diff   = (C.float() - C_ref).abs().flatten()
numel  = diff.numel()
sample = diff if numel <= 8192*8192 else diff[torch.randint(0, numel, (8192*8192,), device="cuda")]

print(f"max  : {sample.max().item():.6e}")
print(f"mean : {sample.mean().item():.6e}")
print(f"p99  : {sample.kthvalue(int(0.99 * sample.numel()))[0].item():.6e}")
print(f"nan  : {torch.isnan(C).any().item()}")

# ── benchmark ─────────────────────────────────────────────────────────────────
print("\n=== Benchmark ===")
for _ in range(20):
    ext.gemm(A, B)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(2000):
    ext.gemm(A, B)
torch.cuda.synchronize()
t1 = time.time()

avg_ms  = (t1 - t0) * 1e3 / 2000
tflops  = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
print(f"avg  : {avg_ms:.4f} ms  |  {tflops:.2f} TFLOP/s")

ms = triton.testing.do_bench(lambda: ext.gemm(A, B), return_mode="median")
print(f"median (triton): {ms:.4f} ms  |  {2.0*M*N*K/(ms*1e-3)/1e12:.2f} TFLOP/s")



# ── torch.matmul reference (cuBLAS via torch) ─────────────────────────────────
print("\n=== torch.matmul reference ===")

for _ in range(20):
    torch.matmul(A, B)
torch.cuda.synchronize()

# which kernel does torch dispatch on this card?
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    torch.matmul(A, B)
    torch.cuda.synchronize()

cuda_events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
if cuda_events:
    kname = max(cuda_events, key=lambda e: e.self_device_time_total).key
    print(f"dispatched kernel: {kname}")
else:
    print("dispatched kernel: (no CUDA event captured)")

ms_torch = triton.testing.do_bench(lambda: torch.matmul(A, B), return_mode="median")
tflops_torch = 2.0 * M * N * K / (ms_torch * 1e-3) / 1e12
print(f"median (triton): {ms_torch:.4f} ms  |  {tflops_torch:.2f} TFLOP/s")

# ── relative ──────────────────────────────────────────────────────────────────
print("\n=== Relative ===")
print(f"ours   : {ms:.4f} ms  |  {2.0*M*N*K/(ms*1e-3)/1e12:.2f} TFLOP/s")
print(f"torch  : {ms_torch:.4f} ms  |  {tflops_torch:.2f} TFLOP/s")
print(f"speedup: {ms_torch/ms:.3f}x  ({(ms_torch/ms - 1)*100:+.1f}%)")
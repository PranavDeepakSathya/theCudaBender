import torch
import time
import json
from pathlib import Path
from torch.utils.cpp_extension import load


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

torch.manual_seed(0)
device = "cuda"

M, N, K = ext.shape()
print(f"\nUsing shape: M={M}, N={N}, K={K}")

# ------------------------------------------------------------
# FP16 inputs
# ------------------------------------------------------------

A = torch.randn((M, K), device=device, dtype=torch.float16).contiguous()

B_rm = torch.randn((N, K), device=device, dtype=torch.float16)
B = B_rm.t()  # column-major

print("\n=== Correctness ===")

torch.cuda.synchronize()
C = ext.gemm(A, B)
torch.cuda.synchronize()

# ------------------------------------------------------------
# Reference in FP32 (always do this)
# ------------------------------------------------------------

C_ref = A.float() @ B.float()

# Cast kernel result to FP32 before diff
diff = (C.float() - C_ref).abs().flatten()

qs = torch.tensor([0.50, 0.90, 0.99, 0.999, 1.00], device="cuda")
vals = torch.quantile(diff, qs).cpu().tolist()

print("\n=== Error Percentiles (abs) ===")
for q, v in zip(qs.cpu().tolist(), vals):
    print(f"{q*100:6.2f}% : {v:.6e}")

torch.set_printoptions(
    threshold=float("inf"),
    linewidth=200,
    precision=1,
    sci_mode=False,
    edgeitems=None,
)


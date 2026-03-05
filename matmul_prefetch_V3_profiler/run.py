import torch
import json
from pathlib import Path
from torch.utils.cpp_extension import load


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

cfg_path = Path(__file__).parent / cfg_file
with open(cfg_path) as f:
    cfg = json.load(f)

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

device = "cuda"

M, N, K = ext.shape()
print(f"\nUsing shape: M={M}, N={N}, K={K}")

A = torch.randn((M, K), device=device, dtype=torch.bfloat16).contiguous()
B_rm = torch.randn((N, K), device=device, dtype=torch.bfloat16).contiguous()
B = B_rm.t()


# =========================
# RUN KERNEL ONCE
# =========================

print("\nLaunching kernel for profiler capture...")

torch.cuda.synchronize()
prof = ext.gemm(A, B)
torch.cuda.synchronize()

prof = prof.cpu().numpy()


# =========================
# BUILD TRACE
# =========================

TAGS = [
    "SETUP",
    "TMA",
    "LDMATRIX",
    "MMA",
    "STORE"
]

events = []

for stream_id, row in enumerate(prof):

    count = int(row[0])

    for i in range(count):

        sm_id = int(row[1 + i*4 + 0])
        tag   = int(row[1 + i*4 + 1])
        start = int(row[1 + i*4 + 2])
        dur   = int(row[1 + i*4 + 3])

        threads_per_block = cfg["WARPS_PER_BLOCK_M"] * cfg["WARPS_PER_BLOCK_N"] * 32

        thread = stream_id % threads_per_block
        block = stream_id // threads_per_block

        events.append(dict(
            name=TAGS[tag],
            ph="X",
            ts=start,
            dur=dur,
            pid=int(sm_id),          # SM lane
            tid=int(thread),           # warp lane
            args={
                "block": int(block),
                "warp": int(thread // 32),
                "sm": int(sm_id),
            }
        ))


# =========================
# NORMALIZE TIMESTAMPS
# =========================

if len(events) > 0:
    offset = min(e["ts"] for e in events)
    for e in events:
        e["ts"] -= offset


trace = dict(traceEvents=events)

trace_path = Path("trace.json")

with open(trace_path, "w") as f:
    json.dump(trace, f)

print("\nTrace written to:", trace_path.resolve())
print("Open with https://ui.perfetto.dev")
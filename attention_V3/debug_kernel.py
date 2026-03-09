import torch
import math
import subprocess
from torch.utils.cpp_extension import load

LOG_FILE = "debug_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")
    print(msg)


# clear old log
open(LOG_FILE, "w").close()

# -----------------------------
# Build extension (DEBUG MODE)
# -----------------------------

ext = load(
    name="attention_warp_debug",
    sources=["export.cu"],
    extra_cuda_cflags=[
        "-arch=sm_120",                 # device debug
        "-lineinfo",          # line numbers for sanitizer
        "-Xptxas=-v",
        "--use_fast_math",
        "-O0"
    ],
    extra_cflags=["-O0"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

# -----------------------------
# Shapes
# -----------------------------

BH, LQ, D, LK = ext.shape()

log(f"BH={BH}")
log(f"LQ={LQ}")
log(f"D={D}")
log(f"LK={LK}")

device = "cuda"

# -----------------------------
# Test seeds
# -----------------------------

thresholds = [1e-3, 1e-2, 5e-2, 1e-1]

for seed in range(1):

    torch.manual_seed(seed)

    Q = torch.randn((BH, LQ, D), device=device, dtype=torch.bfloat16).contiguous()
    K = torch.randn((BH, LK, D), device=device, dtype=torch.bfloat16).contiguous()

    V = torch.randn((BH, D, LK), device=device, dtype=torch.bfloat16).transpose(1,2)

    assert V.stride(1) == 1

    # -----------------------------
    # Kernel
    # -----------------------------

    O = ext.attention(Q, K, V)
    torch.cuda.synchronize()   # VERY IMPORTANT
    # -----------------------------
    # Reference
    # -----------------------------

    S = torch.matmul(Q.float(), K.float().transpose(-1,-2)) / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V.float())

    diff = (O - O_ref).abs()

    log(f"\nSeed {seed}")
    log(f"Max abs error: {diff.max().item()}")
    log(f"Mean abs error: {diff.mean().item()}")

    for t in thresholds:
        count = (diff > t).sum().item()
        log(f">{t}: {count}")


log("\nFinished correctness tests.")
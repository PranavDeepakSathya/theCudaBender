import torch
import itertools
import hashlib
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.cpp_extension import load
from triton.testing import do_bench

# ── search space ──────────────────────────────
SPACE = {
    "ACC_PER_WARP_M":    [2, 4],
    "ACC_PER_WARP_N":    [2, 4],
    "WARP_K_STAGES":     [2, 4],
    "WARPS_PER_BLOCK_M": [2, 4],
    "WARPS_PER_BLOCK_N": [2, 4],
    "BK_STAGES":         [2, 3, 4],
    "GROUP_M":           [2, 4, 8],
    "GROUP_N":           [2, 4, 8],
    "BLOCK_K":           [32, 64],
}

# ── config validation (mirrors config.cuh static_asserts) ────
M, N, K = 8192, 8192, 8192
mma_m, mma_n, mma_k = 16, 8, 16

def is_pow2(x):
    return x > 1 and (x & (x - 1)) == 0

def valid(c):
    apm, apn, wks = c["ACC_PER_WARP_M"], c["ACC_PER_WARP_N"], c["WARP_K_STAGES"]
    wbm, wbn      = c["WARPS_PER_BLOCK_M"], c["WARPS_PER_BLOCK_N"]
    bk            = c["BLOCK_K"]
    gm, gn        = c["GROUP_M"], c["GROUP_N"]

    if not (is_pow2(apm) and is_pow2(apn) and is_pow2(wbm) and is_pow2(wbn) and is_pow2(bk)):
        return False
    if wks > bk // mma_k:                 return False
    if K % bk != 0:                        return False
    if wbm * wbn * 32 >= 1024:             return False
    BM = mma_m * apm * wbm
    BN = mma_n * apn * wbn
    if M % BM != 0 or N % BN != 0:        return False
    GM, GN = M // BM, N // BN
    if GM % gm != 0 or GN % gn != 0:      return False
    if gm > GM or gn > GN:                 return False
    return True

keys   = list(SPACE.keys())
all_combos = list(itertools.product(*SPACE.values()))
combos     = [c for c in all_combos if valid(dict(zip(keys, c)))]
print(f"{len(combos)}/{len(all_combos)} valid configs — compiling in parallel...\n")

# ── parallel compile ──────────────────────────
SRC = str(Path(__file__).parent / "export.cu")

def compile_config(item):
    cfg_vals, keys, src, shape = item
    M_, N_, K_ = shape
    cfg   = dict(zip(keys, cfg_vals))
    flags = [f"-D{k}={v}" for k, v in cfg.items()] + [f"-DGEMM_M={M_}", f"-DGEMM_N={N_}", f"-DGEMM_K={K_}"]
    name  = "gemm_" + hashlib.md5(str((shape, cfg_vals)).encode()).hexdigest()[:8]
    try:
        load(name=name, sources=[src],
             extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math", *flags],
             extra_cflags=["-O3"], extra_ldflags=["-lcuda"], verbose=False)
        return cfg_vals, name, None
    except Exception as e:
        return cfg_vals, name, str(e)

n_workers = os.cpu_count()
work      = [(c, keys, SRC, (M, N, K)) for c in combos]

compiled = []  # (cfg_vals, name)
failed   = 0

with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futs = {pool.submit(compile_config, w): w for w in work}
    for i, fut in enumerate(as_completed(futs), 1):
        cfg_vals, name, err = fut.result()
        if err:
            failed += 1
            print(f"[{i}/{len(combos)}] SKIP {cfg_vals} — {err}")
        else:
            compiled.append((cfg_vals, name))
            print(f"[{i}/{len(combos)}] OK   {cfg_vals}")

print(f"\n{len(compiled)} compiled, {failed} failed — benchmarking...\n")

# ── sequential bench — one subprocess per config to avoid TSS key exhaustion ──
import subprocess, sys

def bench_one(cfg_vals, name):
    cfg   = dict(zip(keys, cfg_vals))
    flags = [f"-D{k}={v}" for k, v in cfg.items()] + [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]
    script = f"""
import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench
ext = load(name={name!r}, sources=[{SRC!r}],
           extra_cuda_cflags=["-arch=sm_120","-O3","--use_fast_math",{",".join(repr(f) for f in flags)}],
           extra_cflags=["-O3"], extra_ldflags=["-lcuda"], verbose=False)
M_, N_, K_ = ext.shape()
A = torch.randn((M_, K_), device="cuda", dtype=torch.bfloat16).contiguous()
B = torch.randn((N_, K_), device="cuda", dtype=torch.bfloat16).t()
ms = do_bench(lambda: ext.gemm(A, B), return_mode="median")
tflops = 2 * M_ * N_ * K_ / (ms * 1e-3) / 1e12
print(f"{{tflops:.4f}} {{ms:.4f}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    tflops, ms = map(float, r.stdout.strip().split())
    return tflops, ms

import json

RESULTS_FILE = Path(__file__).parent / "autotune_results.json"

# resume from previous run if exists
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} results already saved\n")
else:
    results = []

done_names = {r["name"] for r in results}
best = max(results, key=lambda r: r["tflops"], default=None)
best_tflops = best["tflops"] if best else 0.0

for i, (cfg_vals, name) in enumerate(compiled, 1):
    if name in done_names:
        print(f"[{i}/{len(compiled)}] SKIP (cached) {name}")
        continue

    tflops, info = bench_one(cfg_vals, name)
    cfg = dict(zip(keys, cfg_vals))

    if tflops is None:
        print(f"[{i}/{len(compiled)}] SKIP {cfg_vals} — {info}")
        continue

    print(f"[{i}/{len(compiled)}] {cfg_vals}  →  {tflops:.2f} TFLOP/s  ({info:.3f} ms)")

    results.append({"name": name, "cfg": cfg, "tflops": tflops, "ms": info})
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    if tflops > best_tflops:
        best_tflops = tflops

best = max(results, key=lambda r: r["tflops"])
print(f"\n=== Best: {best['tflops']:.2f} TFLOP/s ===")
print(best["cfg"])

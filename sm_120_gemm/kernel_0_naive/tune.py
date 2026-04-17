# k0/tune.py
import torch
import itertools
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.cpp_extension import load

SPACE = {
    "ACC_PER_WARP_M":    [2, 4, 8],
    "ACC_PER_WARP_N":    [2, 4, 8],
    "WARPS_PER_BLOCK_M": [2, 4, 8],
    "WARPS_PER_BLOCK_N": [2, 4, 8],
    "BLOCK_K":           [16, 32, 64, 128],
}

M, N, K     = 8192, 8192, 8192
mma_m, mma_n, mma_k = 16, 8, 16

# ── query hardware limits from the actual GPU ────────────────────────────────
_props = torch.cuda.get_device_properties(0)
MAX_SMEM_PER_BLOCK  = _props.shared_memory_per_block_optin
MAX_REGS_PER_SM     = _props.regs_per_multiprocessor
MAX_REGS_PER_THREAD = 255   # architectural limit, not exposed via torch
MAX_THREADS_PER_SM  = _props.max_threads_per_multi_processor

def is_pow2(x): return x > 0 and (x & (x-1)) == 0

def valid(c):
    apm  = c["ACC_PER_WARP_M"]
    apn  = c["ACC_PER_WARP_N"]
    wbm  = c["WARPS_PER_BLOCK_M"]
    wbn  = c["WARPS_PER_BLOCK_N"]
    bk   = c["BLOCK_K"]

    if not all(is_pow2(x) for x in [apm, apn, wbm, wbn, bk]): return False
    if bk % mma_k != 0:                                         return False
    if K % bk != 0:                                              return False

    # ── tile sizes ────────────────────────────────────────────────────────────
    BM = mma_m * apm * wbm
    BN = mma_n * apn * wbn
    if M % BM != 0 or N % BN != 0:                             return False

    # ── warp_k_iters must be pow2 and >1 (static_assert in config.cuh) ───────
    warp_k_iters = bk // mma_k
    if not is_pow2(warp_k_iters):          return False

    # ── threads per block: config.cuh requires block_size < 1024 ──────────────
    num_warps   = wbm * wbn
    block_size  = num_warps * 32
    if block_size >= 1024:                                       return False

    # ── shared memory ─────────────────────────────────────────────────────────
    As_bytes      = BM * bk * 2        # bf16
    Bs_bytes      = bk * BN * 2        # bf16
    barrier_bytes = 8
    smem_bytes    = As_bytes + Bs_bytes + barrier_bytes
    if smem_bytes > MAX_SMEM_PER_BLOCK:                          return False
    
    # --- TMA block sizes ------ 
    if BM > 256 or BN > 256:                                     return False 

    # ── register pressure estimate ────────────────────────────────────────────
    # ra[apm][4] + rb[apn][2] + rc[apm][apn][4] + ~40 overhead (addresses,
    # loop vars, TMA descriptors, mbarrier state, etc.)
    regs_est = apm * 4 + apn * 2 + apm * apn * 4 + 40
    if regs_est > MAX_REGS_PER_THREAD:                           return False

    # need at least 1 block fitting in SM register file
    if regs_est * block_size > MAX_REGS_PER_SM:                  return False

    return True

SRC  = str(Path(__file__).parent / "export.cu")
keys = list(SPACE.keys())
combos = [c for c in
          [dict(zip(keys, v)) for v in itertools.product(*SPACE.values())]
          if valid(c)]

print(f"{len(combos)} valid configs\n")

# ── parallel compile ──────────────────────────────────────────────────────────
def compile_one(item):
    cfg, src, shape = item
    flags = [f"-D{k}={v}" for k, v in cfg.items()] + \
            [f"-DGEMM_M={shape[0]}", f"-DGEMM_N={shape[1]}", f"-DGEMM_K={shape[2]}"]
    name  = "k0_" + hashlib.md5(str((shape, sorted(cfg.items()))).encode()).hexdigest()[:8]
    try:
        load(name=name, sources=[src],
             extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math", *flags],
             extra_cflags=["-O3"], extra_ldflags=["-lcuda"], verbose=False)
        return cfg, name, None
    except Exception as e:
        return cfg, name, str(e)

compiled, failed = [], 0
work = [(c, SRC, (M, N, K)) for c in combos]

import psutil
_avail_gb = psutil.virtual_memory().available / (1024**3)
_nvcc_est_gb = 6  # conservative per-process estimate for nvcc -O3 on template CUDA
_max_compile = max(1, int(_avail_gb // _nvcc_est_gb))
print(f"available RAM: {_avail_gb:.1f} GB → {_max_compile} parallel compiles\n")

with ProcessPoolExecutor(max_workers=_max_compile) as pool:
    futs = {pool.submit(compile_one, w): w for w in work}
    for i, fut in enumerate(as_completed(futs), 1):
        cfg, name, err = fut.result()
        if err:
            failed += 1
            print(f"[{i}/{len(combos)}] FAIL {cfg} — {err}")
        else:
            compiled.append((cfg, name))
            print(f"[{i}/{len(combos)}] OK   {cfg}")

print(f"\n{len(compiled)} compiled, {failed} failed — benchmarking...\n")

# ── sequential bench ──────────────────────────────────────────────────────────
RESULTS = Path(__file__).parent / "autotune_results.json"
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
done    = {r["name"] for r in results}

for i, (cfg, name) in enumerate(compiled, 1):
    if name in done:
        print(f"[{i}/{len(compiled)}] SKIP (cached) {name}")
        continue

    flags = [f"-D{k}={v}" for k, v in cfg.items()] + \
            [f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]

    script = f"""
import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench
ext = load(name={name!r}, sources=[{SRC!r}],
           extra_cuda_cflags=["-arch=sm_120","-O3","--use_fast_math",{",".join(repr(f) for f in flags)}],
           extra_cflags=["-O3"], extra_ldflags=["-lcuda"], verbose=False)
M, N, K = {M}, {N}, {K}
A = torch.randn((M,K), device="cuda", dtype=torch.bfloat16).contiguous()
B = torch.randn((N,K), device="cuda", dtype=torch.bfloat16).t()
ms = do_bench(lambda: ext.gemm(A,B), return_mode="median")
tflops = 2*M*N*K / (ms*1e-3) / 1e12
print(f"{{tflops:.4f}} {{ms:.4f}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{i}/{len(compiled)}] FAIL {cfg} — {r.stderr.strip()}")
        continue

    tflops, ms = map(float, r.stdout.strip().split())
    print(f"[{i}/{len(compiled)}] {cfg}  →  {tflops:.2f} TFLOP/s  ({ms:.3f} ms)")

    results.append({"name": name, "cfg": cfg, "tflops": tflops, "ms": ms})
    RESULTS.write_text(json.dumps(results, indent=2))

best = max(results, key=lambda r: r["tflops"])
print(f"\n=== best: {best['tflops']:.2f} TFLOP/s ===")
print(best["cfg"])
import torch
import itertools
import hashlib
import os
import ctypes
import shutil
import subprocess
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from triton.testing import do_bench

# ── paths ─────────────────────────────────────
HERE      = Path(__file__).parent.resolve()
BENCH_SRC = str(HERE / "bench.cu")
BUILD_DIR = HERE / "build"
BUILD_DIR.mkdir(exist_ok=True)
RESULTS_FILE = HERE / "autotune_results.json"

NVCC      = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
CUDA_HOME = Path(NVCC).parent.parent
CUDA_INC  = str(CUDA_HOME / "include")
ATOMS_INC = str(HERE.parent / "atoms")

_bench_hash = hashlib.md5(Path(BENCH_SRC).read_bytes()).hexdigest()[:8]

# ── problem size — LLaMA attention proj ───────
# L=4 heads/layers, M=4096 seq*batch, N=4096 out, K=4096 in
# flops per call: 2*L*M*N*K + L*M*N (bias) + L*M*N (silu) ≈ 2*L*M*N*K
L, M, N, K = 4, 4096, 4096, 4096

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

# ── config validation ─────────────────────────
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
    if wks > bk // mma_k:              return False
    if K % bk != 0:                    return False
    if wbm * wbn * 32 >= 1024:         return False
    BM = mma_m * apm * wbm
    BN = mma_n * apn * wbn
    GM = (M + BM - 1) // BM
    GN = (N + BN - 1) // BN
    gm = min(gm, GM)
    gn = min(gn, GN)
    if GM % gm != 0 or GN % gn != 0:  return False
    return True

keys       = list(SPACE.keys())
all_combos = list(itertools.product(*SPACE.values()))
combos     = [c for c in all_combos if valid(dict(zip(keys, c)))]
print(f"{len(combos)}/{len(all_combos)} valid configs\n")

# ── compile ───────────────────────────────────

def cfg_hash(cfg_vals):
    return hashlib.md5(str(((L, M, N, K), cfg_vals, _bench_hash)).encode()).hexdigest()[:12]

def so_path(h):
    return str(BUILD_DIR / f"gemm_{h}.so")

def compile_one(item):
    cfg_vals, keys_, src, shape, cuda_inc, atoms_inc, nvcc_bin, bench_h = item
    L_, M_, N_, K_ = shape
    cfg   = dict(zip(keys_, cfg_vals))
    flags = (
        [f"-D{k}={v}" for k, v in cfg.items()] +
        [f"-DGEMM_L={L_}", f"-DGEMM_M={M_}", f"-DGEMM_N={N_}", f"-DGEMM_K={K_}"]
    )
    h   = hashlib.md5(str((shape, cfg_vals, bench_h)).encode()).hexdigest()[:12]
    out = str(Path(src).parent / "build" / f"gemm_{h}.so")
    if Path(out).exists():
        return cfg_vals, h, None
    cmd = [
        nvcc_bin,
        "-shared", "-Xcompiler", "-fPIC",
        "-arch=sm_120", "-O3", "--use_fast_math",
        f"-I{cuda_inc}", f"-I{atoms_inc}",
        *flags, src, "-o", out, "-lcuda",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return cfg_vals, h, r.stderr.strip()
    return cfg_vals, h, None

n_workers = os.cpu_count()
work = [(c, keys, BENCH_SRC, (L, M, N, K), CUDA_INC, ATOMS_INC, NVCC, _bench_hash) for c in combos]

compiled, failed = [], 0
with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futs = {pool.submit(compile_one, w): w for w in work}
    for i, fut in enumerate(as_completed(futs), 1):
        cfg_vals, h, err = fut.result()
        if err:
            failed += 1
            print(f"[{i}/{len(combos)}] FAIL {cfg_vals}\n  {err[:120]}")
        else:
            compiled.append((cfg_vals, h))
            print(f"[{i}/{len(combos)}] OK   {cfg_vals}")

print(f"\n{len(compiled)} compiled, {failed} failed — benchmarking...\n")

# ── bench ─────────────────────────────────────
results     = []
done_hashes = set()
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    done_hashes = {r["hash"] for r in results}
    print(f"Resuming — {len(results)} already benchmarked\n")

TensorMap = ctypes.c_uint8 * 128

torch.manual_seed(0)
device = "cuda"
A    = torch.randn((L, M, K), device=device, dtype=torch.bfloat16).contiguous()
B    = torch.randn((L, N, K), device=device, dtype=torch.bfloat16).transpose(-2, -1).contiguous()
C    = torch.empty((L, M, N), device=device, dtype=torch.float32)
bias = torch.randn((N,),      device=device, dtype=torch.float32)

A_ptr    = ctypes.c_void_p(A.data_ptr())
B_ptr    = ctypes.c_void_p(B.data_ptr())
C_ptr    = ctypes.c_void_p(C.data_ptr())
bias_ptr = ctypes.c_void_p(bias.data_ptr())

for i, (cfg_vals, h) in enumerate(compiled, 1):
    if h in done_hashes:
        print(f"[{i}/{len(compiled)}] cached {h}")
        continue

    lib = ctypes.CDLL(so_path(h))
    lib.build_tma_maps.restype  = None
    lib.build_tma_maps.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p]
    lib.init_kernel.restype     = ctypes.c_int
    lib.init_kernel.argtypes    = []
    lib.run_gemm.restype        = ctypes.c_int
    lib.run_gemm.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    a_map, b_map = TensorMap(), TensorMap()
    lib.build_tma_maps(A_ptr, B_ptr,
                       ctypes.cast(a_map, ctypes.c_void_p),
                       ctypes.cast(b_map, ctypes.c_void_p))
    a_ct = ctypes.cast(a_map, ctypes.c_void_p)
    b_ct = ctypes.cast(b_map, ctypes.c_void_p)

    if lib.init_kernel() != 0:
        print(f"[{i}/{len(compiled)}] SKIP {cfg_vals} — invalid launch config")
        continue

    try:
        ms = do_bench(
            lambda: lib.run_gemm(A_ptr, B_ptr, C_ptr, bias_ptr, a_ct, b_ct),
            return_mode="median"
        )
    except Exception as e:
        print(f"[{i}/{len(compiled)}] SKIP {cfg_vals} — {e}")
        continue

    flops  = 2 * L * M * N * K + L * M * N
    tflops = flops / (ms * 1e-3) / 1e12
    cfg    = dict(zip(keys, cfg_vals))
    print(f"[{i}/{len(compiled)}] {cfg_vals}  →  {tflops:.2f} TFLOP/s  ({ms:.3f} ms)")

    results.append({"hash": h, "cfg": cfg, "tflops": tflops, "ms": ms})
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    done_hashes.add(h)

if results:
    best = max(results, key=lambda r: r["tflops"])
    print(f"\n=== Best: {best['tflops']:.2f} TFLOP/s ===")
    print(best["cfg"])

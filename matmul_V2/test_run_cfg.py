import json
import subprocess
import sys

CFG_FILE = sys.argv[1] if len(sys.argv) > 1 else "5090_cfg.json"


with open(CFG_FILE) as f:
    cfg = json.load(f)

print("\n=== Loaded Config ===")
for k, v in cfg.items():
    print(f"{k} = {v}")


defines = []
for k, v in cfg.items():
    defines.append(f"-D{k}={v}")


cmd = [
    "nvcc",
    "-O3",
    "--use_fast_math",
    "-arch=sm_120",
    *defines,
    "test.cu",
    "-o", "matmul_bench.out",
    "-lcuda",
    "-lcudart"
]

print("\n=== Compiling ===")
print(" ".join(cmd))

subprocess.check_call(cmd)

print("\n=== Running Benchmark ===")
subprocess.check_call(["./matmul_bench.out"])

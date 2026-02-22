import json
import subprocess
import sys
import torch
from pathlib import Path

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

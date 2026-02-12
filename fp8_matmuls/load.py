import torch
from torch.utils.cpp_extension import load

ext = load(
    name="try_ext",
    sources=["try.cu"],
    extra_cuda_cflags=["-arch=sm_120"],
    verbose=True,
)

print("Extension built successfully")
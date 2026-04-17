import torch

A = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
B = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)

# warmup
for _ in range(3): torch.matmul(A, B)
torch.cuda.synchronize()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    torch.matmul(A, B)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
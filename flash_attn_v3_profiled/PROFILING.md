# Flash Attention V3 - Intra-Kernel Profiling Guide

This is a profiled copy of `flash_attn_v3/` instrumented with [Intra-Kernel Profiler (IKP)](https://github.com/yao-jz/intra-kernel-profiler) trace markers. The original kernel is untouched in `../flash_attn_v3/`.

## What IKP Trace Does

IKP's trace backend records **nanosecond-precision timestamps per warp** into lock-free circular ring buffers on the GPU. Each warp leader (lane 0) captures begin/end events using the hardware `%%globaltimer` register. No CUPTI, no NVBit, no PC sampling required -- just CUDA toolkit + C++17.

This gives you **per-region timing breakdowns** of your kernel, answering "which phase is slow?" rather than just "the kernel is slow."

## Profiled Regions

Six regions are instrumented in the flash attention kernel:

| ID | Name | What It Measures |
|----|------|-----------------|
| 1 | `total` | Entire kernel (init to output write) |
| 2 | `load_q` | TMA load Q to shared memory + ldmatrix to registers |
| 3 | `load_k` | Wait K barrier + ldmatrix K^T to registers |
| 4 | `compute_qk` | Q @ K^T matrix multiply-accumulate (MMA) + scale |
| 5 | `softmax` | Online softmax update + pack scores to bf16 |
| 6 | `compute_pv` | Wait V barrier + ldmatrix V + P @ V MMA |

The main loop body (regions 3-6) runs `L_kv / block_L_kv = 64` iterations per block. Regions 3-6 are measured inside every iteration.

## Quick Start

```bash
# Activate the project venv
source /root/theCudaBender/.venv/bin/activate

# Run (builds automatically via torch cpp_extension)
cd /root/theCudaBender/flash_attn_v3_profiled
python run.py
```

This produces:
- `flash_attn_v3_trace.json` -- Chrome Trace JSON (viewable in Perfetto)
- `flash_attn_v3_trace_summary.json` -- Per-region statistics (mean, percentiles, histograms)

## Viewing the Trace

### Perfetto (recommended)
1. Open https://ui.perfetto.dev in your browser
2. Click "Open trace file" and select `flash_attn_v3_trace.json`
3. You'll see a timeline grouped by SM (process) and warp (thread)
4. Zoom into a single block to see the pipeline stages

### Reading the Summary JSON

```python
import json

with open("flash_attn_v3_trace_summary.json") as f:
    d = json.load(f)

for r in d["regions"]:
    p = r["percentiles"]
    print(f'{r["name"]:<15} count={r["count"]:>8}  '
          f'mean={r["mean_dur"]:>8.0f}ns  '
          f'p50={p["p50"]:>8.0f}ns  '
          f'p95={p["p95"]:>8.0f}ns')
```

### Key Metrics to Look At

- **mean_dur**: Average duration in nanoseconds (raw globaltimer ticks = ns)
- **cv_dur**: Coefficient of variation -- high CV means high variance across warps (load imbalance)
- **p50 vs p95**: If p95 >> p50, some warps are stalling (likely barrier waits)
- **unmatched_begin/end**: Should be 0. Non-zero means a bug in instrumentation
- **count**: Number of begin/end pairs. `total` = num_blocks * warps_per_block. Loop regions = that * loop_iterations

## How the Instrumentation Works

### Device Side (kernel.cuh)

```cuda
#include <intra_kernel_profiler/trace/trace.cuh>

// Define a trace context type: 32768 events per warp, 8 warps per block
using TraceCtx = IKP_TRACE_CTX_TYPE(32768, Cfg::num_warps);
TraceCtx ikp_ctx;

// Initialize (once, after __syncthreads following barrier init)
IKP_TRACE_CTX_INIT(ikp_ctx);

// Record begin/end around code regions
IKP_TRACE_REC_B(ikp_ctx, ikp_buf, region_id);  // begin
// ... code to measure ...
IKP_TRACE_REC_E(ikp_ctx, ikp_buf, region_id);  // end

// Flush at kernel end (before output writes)
IKP_TRACE_CTX_FLUSH(ikp_ctx, ikp_buf);
```

The kernel receives `ikp::GlobalBuffer ikp_buf` as an extra parameter (passed by value, it's just two pointers).

### Host Side (export.cu)

```cpp
#include <intra_kernel_profiler/trace/trace.cuh>
namespace ikp = intra_kernel_profiler::trace;

// Global session (persists across calls)
static ikp::HostSession g_ikp_session;

// Init once
g_ikp_session.set_region_names({"_unused", "total", "load_q", ...});
g_ikp_session.init(32768, grid_size, block_size);

// Before each profiled launch
g_ikp_session.reset();

// Launch kernel with g_ikp_session.global_buffer() as extra arg
launch_attention<Cfg>(launcher, q_map, k_map, v_map, O_ptr,
                      g_ikp_session.global_buffer());

// After kernel completes
cudaDeviceSynchronize();
ikp::TraceWriteOptions opt;
opt.scale = 1.0;  // 1.0 = raw ns
opt.emit_summary_json = true;
g_ikp_session.write_trace("flash_attn_v3_trace.json", opt);
```

### Python Side (run.py)

The `dump_trace(path)` function is exposed via pybind11:
```python
_ = ext.attention(Q, K, V)  # profiled run
ext.dump_trace("flash_attn_v3_trace.json")
```

### Build

Just add the IKP include path to your CUDA flags:
```python
extra_cuda_cflags=[
    "-arch=sm_120",
    "-O3",
    "--use_fast_math",
    "-I/root/intra-kernel-profiler/include",
]
```

IKP trace is header-only -- no libraries to link.

## Adding / Modifying Regions

1. Define new region IDs in the `ProfileRegion` enum in `kernel.cuh` (max 7 regions, IDs 1-7)
2. Wrap code with `IKP_TRACE_REC_B` / `IKP_TRACE_REC_E`
3. Add the name to `set_region_names()` in `export.cu` (index = region ID)
4. Rebuild and run

### Conditional Recording (for hot loops)

If you want to sample only every Nth iteration to reduce overhead:
```cuda
const uint32_t sample_mask = (1u << 4) - 1;  // every 16 iterations
IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, region_id, 0, (iter & sample_mask) == 0);  // begin
// ... code ...
IKP_TRACE_REC_IF(ikp_ctx, ikp_buf, region_id, 1, (iter & sample_mask) == 0);  // end
```

### Block Filtering

To trace only specific blocks (reduces trace size):
```cpp
g_ikp_session.set_block_filter({0, 1, 2, 3});  // only blocks 0-3
```

## Overhead

- ~1-2% per recorded event (streaming store bypasses caches)
- Each event = 16 bytes (timestamp + region ID + block/warp info)
- Buffer: 32768 events * 16 bytes * 8 warps * 2048 blocks = ~8 GB allocated
- To reduce memory: lower the per-warp capacity (e.g., 4096) or use block filtering
- The profiled kernel runs at the same TFLOP/s as the original (177 TFLOP/s)

## TraceWriteOptions

| Option | Default | Description |
|--------|---------|-------------|
| `scale` | 1.0 | Multiply timestamps by this. 1.0 = raw ns. Use 1e-3 for microseconds |
| `emit_complete_events` | true | Pair begin/end into Chrome "X" (complete) events |
| `group_by_smid` | true | Group by SM in Perfetto. false = group by block |
| `emit_summary_json` | true | Write `*_summary.json` with stats |
| `summary_hist_bins` | 128 | Histogram resolution |
| `emit_block_region_distributions` | false | Extra per-block breakdown files |

## File Overview

```
flash_attn_v3_profiled/
  config.cuh     -- Unchanged config constants
  kernel.cuh     -- Instrumented kernel (IKP trace markers added)
  export.cu      -- Host code with IKP session + dump_trace() binding
  run.py         -- Build + run + trace dump
  PROFILING.md   -- This file
```

## Troubleshooting

**Large trace file**: The trace JSON can be 500MB+. Use block filtering or reduce capacity.

**Perfetto slow to load**: Filter to fewer blocks, or use `group_by_smid=false` for fewer tracks.

**unmatched_begin > 0**: A begin event has no matching end. Check that every `REC_B` has a `REC_E` on the same code path (beware early returns or divergent branches).

**Zero events**: Make sure `IKP_TRACE_CTX_INIT` runs before any `REC` calls, and `IKP_TRACE_CTX_FLUSH` runs before the kernel exits.

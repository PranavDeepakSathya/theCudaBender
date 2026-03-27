"""
Drop-in Perfetto trace exporter for GPU kernel profiler.

Usage:
    from profiler.trace import export_trace

    # after kernel returns the profiler tensor:
    export_trace(
        profiler_tensor,             # int64 tensor [num_streams, 1+max_events*4]
        tags=["SETUP", "TMA", "LDMATRIX", "MMA", "STORE"],
        streams_per_block=4,         # WARPS_PER_BLOCK or THREADS_PER_BLOCK
        granularity="warp",          # "warp" or "thread"
        output="trace.json",
    )

    # then open trace.json at https://ui.perfetto.dev
"""

import json
import numpy as np
from pathlib import Path


def export_trace(
    profiler,                   # numpy array or torch tensor [num_streams, cols]
    tags: list[str],            # tag index -> name
    streams_per_block: int,     # warps or threads per block
    granularity: str = "warp",  # "warp" | "thread"
    output: str = "trace.json",
):
    """
    Convert raw profiler buffer to Chrome/Perfetto trace JSON.

    Args:
        profiler:          2D int64 array, shape [num_streams, 1 + max_events*4].
                           Pass a torch tensor (will be moved to CPU) or numpy array.
        tags:              List of tag names. Index must match the tag ints used in
                           start(tag) calls on the device side.
        streams_per_block: Number of profiling streams per block.
                           - warp-level:  WARPS_PER_BLOCK
                           - thread-level: THREADS_PER_BLOCK (= warps * 32)
        granularity:       "warp" or "thread". Controls how tid is assigned in the
                           trace (warp id vs thread id within block).
        output:            Path for the output trace JSON file.
    """
    # handle torch tensors
    if hasattr(profiler, 'cpu'):
        profiler = profiler.cpu().numpy()

    assert profiler.ndim == 2, f"Expected 2D array, got {profiler.ndim}D"

    events = []

    for stream_id in range(profiler.shape[0]):
        row = profiler[stream_id]
        count = int(row[0])

        if count == 0:
            continue

        block = stream_id // streams_per_block
        local = stream_id % streams_per_block  # warp or thread within block

        if granularity == "warp":
            tid = local
            warp = local
        else:
            tid = local
            warp = local // 32

        for i in range(count):
            base   = 1 + i * 4
            sm_id  = int(row[base + 0])
            tag    = int(row[base + 1])
            start  = int(row[base + 2])
            dur    = int(row[base + 3])

            tag_name = tags[tag] if tag < len(tags) else f"TAG_{tag}"

            events.append({
                "name": tag_name,
                "ph":   "X",
                "ts":   start,
                "dur":  dur,
                "pid":  sm_id,
                "tid":  tid,
                "args": {
                    "block": block,
                    "warp":  warp,
                    "sm":    sm_id,
                },
            })

    # normalize timestamps to start at 0
    if events:
        t_min = min(e["ts"] for e in events)
        for e in events:
            e["ts"] -= t_min

    trace = {"traceEvents": events}

    out_path = Path(output)
    with open(out_path, "w") as f:
        json.dump(trace, f)

    print(f"Trace written to: {out_path.resolve()}")
    print(f"  {len(events)} events from {profiler.shape[0]} streams")
    print(f"  Open with https://ui.perfetto.dev")

    return out_path

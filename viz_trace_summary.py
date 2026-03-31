"""
Generic IKP trace summary visualizer.
Usage:
    python viz_trace_summary.py <summary.json> [--out <prefix>]

Produces:
  1. Horizontal bar chart of mean durations per region
  2. Per-region percentile box/whisker chart
  3. Per-warp breakdown for a selected region (if by_block_warp data exists)
"""

import json
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_summary(path):
    with open(path) as f:
        return json.load(f)

def plot_bar(regions, out_prefix):
    names = [r["name"] for r in regions if r["name"] != "total"]
    means = [r["mean_dur"] for r in regions if r["name"] != "total"]
    cvs   = [r["cv_dur"] for r in regions if r["name"] != "total"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    bars = ax.barh(names, means, color="#4C72B0", edgecolor="white")

    for bar, cv in zip(bars, cvs):
        w = bar.get_width()
        ax.text(w + max(means)*0.01, bar.get_y() + bar.get_height()/2,
                f"cv={cv:.3f}", va="center", fontsize=8, color="#555")

    ax.set_xlabel("Mean duration (ns)")
    ax.set_title("Per-region mean duration")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", dpi=150)
    print(f"Saved {out_prefix}_bar.png")

def plot_percentiles(regions, out_prefix):
    names = [r["name"] for r in regions if r["name"] != "total"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))

    for i, r in enumerate(regions):
        if r["name"] == "total":
            continue
        p = r["percentiles"]
        ax.barh(r["name"], p["p95"] - p["p5"], left=p["p5"],
                height=0.6, color="#4C72B0", alpha=0.3, edgecolor="none")
        ax.barh(r["name"], p["p75"] - p["p25"], left=p["p25"],
                height=0.6, color="#4C72B0", alpha=0.6, edgecolor="none")
        ax.plot(p["p50"], r["name"], "w|", markersize=12, markeredgewidth=2)
        ax.plot(r["mean_dur"], r["name"], "r+", markersize=8, markeredgewidth=1.5)

    ax.set_xlabel("Duration (ns)")
    ax.set_title("Per-region distribution  (dark=IQR, light=p5-p95, |=median, +=mean)")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pct.png", dpi=150)
    print(f"Saved {out_prefix}_pct.png")

def plot_warp_breakdown(summary, out_prefix):
    bw = summary.get("by_block_warp_regions", {})
    if not bw:
        return

    region_names = []
    warp_data = {}  # region_name -> {warp_id: mean_dur}

    for key, val in bw.items():
        name = val["name"]
        if name == "total":
            continue
        region_names.append(name)
        warp_data[name] = {}
        for entry in val["by_block_warp"]:
            warp_data[name][entry["warp"]] = entry["mean_dur"]

    if not region_names:
        return

    all_warps = sorted(set(w for d in warp_data.values() for w in d))
    n_warps = len(all_warps)

    fig, ax = plt.subplots(figsize=(12, max(4, len(region_names) * 0.6)))
    bar_height = 0.8 / n_warps
    colors = plt.cm.tab10(np.linspace(0, 1, n_warps))

    for wi, warp_id in enumerate(all_warps):
        vals = [warp_data[name].get(warp_id, 0) for name in region_names]
        positions = np.arange(len(region_names)) + wi * bar_height - 0.4
        ax.barh(positions, vals, height=bar_height, color=colors[wi],
                label=f"warp {warp_id}", edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels(region_names)
    ax.set_xlabel("Mean duration (ns)")
    ax.set_title("Per-warp breakdown (block 0)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_warps.png", dpi=150)
    print(f"Saved {out_prefix}_warps.png")

def main():
    parser = argparse.ArgumentParser(description="Visualize IKP trace summary")
    parser.add_argument("summary_json", help="Path to *_trace_summary.json")
    parser.add_argument("--out", default=None, help="Output file prefix (default: derived from input)")
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    out_prefix = args.out or args.summary_json.replace("_summary.json", "").replace(".json", "")

    regions = summary["regions"]
    plot_bar(regions, out_prefix)
    plot_percentiles(regions, out_prefix)
    plot_warp_breakdown(summary, out_prefix)

if __name__ == "__main__":
    main()

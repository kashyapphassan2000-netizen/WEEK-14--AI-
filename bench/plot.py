"""Plot latency vs seqlen from bench/results/results_fwd.json.

    python bench/run.py          # produces the JSON
    python bench/plot.py         # produces bench/results/latency.png
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(RESULTS_DIR, "results_fwd.json"))
    ap.add_argument("--output", default=os.path.join(RESULTS_DIR, "latency.png"))
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    meta, rows = data["meta"], data["rows"]
    xs = [r["seqlen"] for r in rows]

    plt.figure(figsize=(7, 4.5))
    for key, style, label in (
        ("naive_ms", "o-", "naive Triton (materializes N x N)"),
        ("ours_ms", "s-", "ours (flash)"),
        ("torch_ms", "^-", "torch SDPA"),
    ):
        ys = [r[key] for r in rows]
        if all(math.isnan(y) for y in ys):
            continue
        plt.plot(xs, ys, style, label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length")
    plt.ylabel("latency (ms, lower is better)")
    plt.title(f"FlashAttention v1 in Triton, {meta['mode']}, fp16, "
              f"B={meta['B']} H={meta['H']} D={meta['D']}\n{meta['gpu']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

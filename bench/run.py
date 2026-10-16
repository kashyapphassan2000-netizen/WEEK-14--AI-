"""Day 6: latency of naive Triton vs our flash kernel vs torch SDPA.

    python bench/run.py --seqlen 1024,2048,4096,8192
    python bench/run.py --mode fwd_bwd          # training step (no naive: it has no backward)

Writes bench/results/results_<mode>.json and prints a markdown table you can
paste into the README. Needs a CUDA GPU; the CPU interpreter is for
correctness only and its timings mean nothing.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton  # noqa: E402

from kernels.attn_bwd import flash_attention  # noqa: E402
from kernels.attn_fwd import flash_attn_fwd  # noqa: E402
from kernels.attn_naive import naive_attention  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def bench_ms(fn):
    """Median ms per call. do_bench handles warmup and flushes L2 between runs."""
    try:
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")


def attn_tflops(B, H, N, D, ms, causal, mode):
    flops = 4 * B * H * N * N * D  # two matmuls of 2*N*N*D each
    if causal:
        flops //= 2
    if mode == "fwd_bwd":
        flops = int(flops * 3.5)  # bwd = 2.5x fwd (5 matmuls vs 2)
    return flops / (ms * 1e-3) / 1e12


def run(seqlens, B=1, H=16, D=64, causal=False, mode="fwd", dtype=torch.float16):
    rows = []
    for N in seqlens:
        torch.manual_seed(0)
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=(mode == "fwd_bwd"))
                   for _ in range(3)]
        if mode == "fwd":
            t_naive = bench_ms(lambda: naive_attention(q, k, v, causal=causal))
            t_ours = bench_ms(lambda: flash_attn_fwd(q, k, v, causal=causal))
            t_torch = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        else:
            do = torch.randn_like(q)
            t_naive = float("nan")
            t_ours = bench_ms(lambda: flash_attention(q, k, v, causal=causal).backward(do))
            t_torch = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal).backward(do))
        rows.append({
            "seqlen": N, "naive_ms": t_naive, "ours_ms": t_ours, "torch_ms": t_torch,
            "speedup_vs_naive": t_naive / t_ours,
            "ours_vs_torch": t_ours / t_torch,
            "ours_tflops": attn_tflops(B, H, N, D, t_ours, causal, mode),
        })
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def markdown_table(rows):
    lines = [
        "| seqlen | naive Triton (ms) | this kernel (ms) | torch SDPA (ms) | speedup vs naive | ours / SDPA | TFLOP/s |",
        "|-------:|------------------:|-----------------:|----------------:|-----------------:|------------:|--------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seqlen']} | {r['naive_ms']:.3f} | {r['ours_ms']:.3f} | {r['torch_ms']:.3f} "
            f"| {r['speedup_vs_naive']:.2f}x | {r['ours_vs_torch']:.2f}x | {r['ours_tflops']:.1f} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=str, default="1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench/run.py needs a CUDA GPU (see README: Kaggle T4 is free).")

    seqlens = [int(x) for x in args.seqlen.split(",")]
    rows = run(seqlens, args.batch, args.heads, args.head_dim, args.causal, args.mode)

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "triton": triton.__version__,
        "B": args.batch, "H": args.heads, "D": args.head_dim,
        "causal": args.causal, "mode": args.mode, "dtype": "float16",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_{args.mode}{'_causal' if args.causal else ''}.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)

    print(f"{meta['gpu']} | B={args.batch} H={args.heads} D={args.head_dim} "
          f"causal={args.causal} mode={args.mode} fp16\n")
    print(markdown_table(rows))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

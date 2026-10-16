# flash-att-triton

FlashAttention v1 (Dao et al., 2022) implemented from scratch in Triton: forward
kernel with online-softmax tiling, backward pass by recomputation, a naive Triton
baseline, and benchmarks against `torch.nn.functional.scaled_dot_product_attention`.

- **Forward:** one pass over K/V tiles, running max and sum in registers, the N × N
  matrix never touches HBM, `exp2` fast path, causal masking, any sequence length.
- **Backward:** three kernels (`D = rowsum(O·dO)`, dK/dV, dQ) that rebuild P from the
  saved log-sum-exp. No atomics. Wrapped in `torch.autograd.Function`.
- **Tested** against fp32 SDPA for outputs, LSE, and dQ/dK/dV, causal and non-causal,
  head dims 16–128, ragged N, strided inputs. Runs on GPU, or on CPU through the
  Triton interpreter (CI does this on every push).

New here? Start with **[NOTES.md](NOTES.md)**: a complete beginner's walkthrough from
creating the virtual environment to tagging the release, with every term explained and an
honest status of what is and isn't verified. To build the whole thing yourself from an
empty folder, file by file with every piece of code explained, follow
**[NOTES_IMPLEMENTATION.md](NOTES_IMPLEMENTATION.md)**. Then read
**[docs/decisions.md](docs/decisions.md)** for why each design choice was made.

## Quickstart

```bash
pip install -r requirements.txt

# Tests: real kernels on a CUDA GPU; on a CPU box they run in the Triton interpreter
pytest tests/ -v

# Benchmarks (CUDA GPU required)
python bench/run.py --seqlen 1024,2048,4096,8192            # forward
python bench/run.py --seqlen 1024,2048,4096,8192 --mode fwd_bwd
python bench/plot.py                                         # -> bench/results/latency.png

# Or all of the above in one go on a GPU box
bash scripts/run_all_gpu.sh
```

```python
import torch
from kernels import flash_attention

q, k, v = (torch.randn(2, 16, 4096, 64, device="cuda", dtype=torch.float16, requires_grad=True)
           for _ in range(3))
o = flash_attention(q, k, v, causal=True)   # same as F.scaled_dot_product_attention(..., is_causal=True)
o.sum().backward()
```

Supported: fp16 / bf16, head dim ∈ {16, 32, 64, 128}, any sequence length, causal or not.

## Results

> **Not measured yet.** This repo was built and correctness-tested without a GPU
> (Triton CPU interpreter). Run `bash scripts/run_all_gpu.sh` on a GPU and paste the
> printed table here. A free Kaggle T4 is enough; an A100 gives the headline numbers.

Forward, fp16, B=1, H=16, D=64, GPU: _TBD_

| seqlen | naive Triton (ms) | this kernel (ms) | torch SDPA (ms) | speedup vs naive | ours / SDPA | TFLOP/s |
|-------:|------------------:|-----------------:|----------------:|-----------------:|------------:|--------:|
| 1024 | | | | | | |
| 2048 | | | | | | |
| 4096 | | | | | | |
| 8192 | | | | | | |

<!-- ![latency](bench/results/latency.png) -->

## Correctness status

| Check | CPU interpreter | GPU |
|---|---|---|
| matmul vs torch | ✅ pass | run `pytest` |
| naive attention vs SDPA (causal + non-causal) | ✅ pass | run `pytest` |
| flash forward vs SDPA, rtol=atol=1e-2 | ✅ pass | run `pytest` |
| LSE vs `torch.logsumexp` | ✅ pass | run `pytest` |
| dQ, dK, dV vs SDPA, rtol=atol=2e-2 | ✅ pass | run `pytest` |
| bf16 | skipped (interpreter `tl.dot` is wrong for bf16) | run `pytest` on Ampere+ (A100, RTX 30xx/40xx); skipped on T4 |
| PASS gate: forward at N=2048, D=64 | skipped (GPU size) | run `pytest` |

## Layout

```
kernels/matmul.py      tiled fp16 matmul (Triton warm-up)
kernels/attn_naive.py  standard attention, materializes N x N in HBM (baseline)
kernels/attn_fwd.py    FlashAttention forward
kernels/attn_bwd.py    FlashAttention backward + autograd wrapper
tests/                 correctness tests, GPU or CPU interpreter
bench/                 benchmark + plot
docs/notes.md          the IO math and online-softmax derivation
docs/decisions.md      plain-English build log and design decisions
blog/post.md           write-up
```

## Reference

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

The kernel structure (exp2 trick, split dK/dV and dQ backward) follows the
[Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

## License

MIT

# FlashAttention v1 in Triton, from scratch

*Forward and backward, tested against PyTorch, with the IO math derived by hand.*

## 1. Why I built this

Attention is the most expensive operation in a transformer, and FlashAttention
(Dao et al., 2022) is why long-context models are practical. It doesn't approximate
anything. It computes *exact* attention, just with far less memory traffic. Its
one idea, keeping the work on-chip and never writing the N × N matrix, changed how
people write GPU kernels.

Reading the paper is one thing. Writing the kernel forces you to get every detail
right: the running max, the rescale, the masking at tile edges, the backward pass
that rebuilds what the forward threw away. So I wrote it in Triton, end to end:

- a naive Triton baseline that materializes the N × N matrix,
- the flash forward kernel with online softmax,
- the backward pass by recomputation,
- tests against PyTorch's `scaled_dot_product_attention`, and benchmarks.

Code: this repo. Build log with every decision: `docs/decisions.md`.

## 2. The IO problem: memory is the bottleneck, not FLOPs

A GPU has a large, slow main memory (HBM, 40–80 GB) and a tiny, fast on-chip memory
(SRAM, about 192 KB per streaming multiprocessor on an A100). Compute is far faster
than HBM can feed it. An A100 does about 312 fp16 TFLOP/s but moves only about
1.5–2 TB/s.

Standard attention for one head:

```
S = Q K^T / sqrt(d)      # N x N, written to HBM
P = softmax(S)           # N x N, read + written
O = P V                  # N x N, read again
```

At N = 4096 in fp16, S is 32 MB per head. It can't stay on-chip, so it makes three
round trips through HBM, and softmax, an elementwise op, is purely memory-bound.
HBM traffic is Θ(N² + Nd). For large N, the N² term is all that matters.

FlashAttention cuts this to Θ(N²d²/M), where M is the SRAM size. The N × N matrix
is never written at all. Only Q, K, V and O, each N × d, move through HBM.

## 3. The online softmax recurrence

Softmax seems to need the whole row: you need the max for stability and the sum for
normalization. The key fact is that both can be computed in pieces and merged.

For one row, split the keys into blocks. After processing some blocks you hold:

- `m`: max score so far
- `l`: sum of `exp(score − m)` so far
- `acc`: sum of `exp(score − m) · v` so far

A new block with scores `s_j` and values `V_j` arrives:

```
m_new = max(m, max(s_j))
p_j   = exp(s_j - m_new)
alpha = exp(m - m_new)
l     = alpha * l + sum(p_j)
acc   = alpha * acc + p_j @ V_j
m     = m_new
```

Why it's exact: `exp(s − m_old) · exp(m_old − m_new) = exp(s − m_new)`. Multiplying
the old state by `alpha` re-expresses every past term relative to the new max. After
the last block, `acc / l` is exactly `softmax(s) @ V`, the same number standard
attention computes, up to floating-point rounding.

One more value falls out for free: `LSE = m + log(l)`, the log-sum-exp of the row.
The backward pass needs only this to rebuild any tile of P:

```
P_ij = exp(S_ij - LSE_i)
```

**The backward pass**, given dO:

```
D   = rowsum(dO * O)
dV  = P^T dO
dP  = dO V^T
dS  = P * (dP - D)        # softmax Jacobian
dQ  = dS K * scale
dK  = dS^T Q * scale
```

`D` comes from `sum_k P_k dP_k = dO · (sum_k P_k V_k) = dO · O`. Storing P would cost
N² memory. Recomputing it costs a matmul per tile. Compute is cheap and memory
traffic is not, so recompute wins.

## 4. Triton tiles and the forward kernel

Each Triton program owns `BLOCK_M` query rows for one (batch, head):

```python
q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
m_i = -inf; l_i = 0; acc = 0
for start_n in range(0, hi, BLOCK_N):
    kT  = tl.load(...)                     # [D, BLOCK_N]
    qk  = tl.dot(q, kT) * qk_scale         # scale * log2(e)
    qk  = tl.where(valid, qk, -inf)        # ragged N + causal mask
    m_ij  = tl.maximum(m_i, tl.max(qk, 1))
    p     = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_i   = l_i * alpha + tl.sum(p, 1)
    acc   = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
    m_i   = m_ij
acc = acc / l_i[:, None]
```

Details that mattered:

- **exp2.** Folding `log2(e)` into the scale lets every exponential use the hardware
  `exp2`. The saved LSE is then in base 2, and the backward uses the same convention.
- **fp32 accumulators.** Q, K, V are fp16. `m`, `l`, `acc` stay fp32 and are cast once
  on store.
- **Masks everywhere.** N doesn't have to be a multiple of the tile size. Out-of-range
  keys get score −∞, so they contribute exp2(−∞) = 0, and out-of-range rows are never
  stored.
- **Causal.** Tiles entirely in the future are skipped (`hi = (start_m+1)·BLOCK_M`),
  and the diagonal tile is masked with `row >= col`. Skipping alone isn't enough: inside
  the diagonal tile, early rows would otherwise see later tokens.
- **int64 offsets** for batch/head strides. At large B·H·N·D, int32 addresses wrap.

The backward uses three kernels: a preprocess for `D`, one kernel where each program
owns a K/V tile and loops over queries (dK, dV), and one where each program owns a Q
tile and loops over keys (dQ). Every output has one writer, so no atomics are needed.
The price is recomputing `QK^T` twice.

**Testing.** Every kernel is checked against fp32 PyTorch SDPA: outputs at 1e-2,
gradients at 2e-2, LSE against `torch.logsumexp`, causal and not, head dims 16–128,
ragged N such as 77 and 1000, and strided inputs. The same tests run on CPU through
Triton's interpreter (`TRITON_INTERPRET=1`), so CI checks correctness without a GPU.
To make sure the tests have teeth, I planted two classic bugs: forgetting `sm_scale`
on dQ, and not rescaling `acc` by `alpha`. Both made the suites fail.

## 5. Benchmark results

> **To be filled in from a GPU run.** `bash scripts/run_all_gpu.sh` prints this table
> and writes `bench/results/latency.png`. No numbers are reported until they are measured.

Forward, fp16, B=1, H=16, D=64, GPU: _TBD_

| seqlen | naive Triton (ms) | this kernel (ms) | torch SDPA (ms) | speedup vs naive | ours / SDPA |
|-------:|------------------:|-----------------:|----------------:|-----------------:|------------:|
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |
| 8192 | | | | | |

What to expect: the gap to naive should widen with N, because naive's HBM traffic
grows as N² while ours grows as N·d per query tile. Matching SDPA within some margin is
a good outcome. On Ampere and newer, SDPA dispatches to hand-tuned FlashAttention-2 CUDA
kernels, which reorder the loops to cut non-matmul work and split work across warps
more efficiently than this v1-style kernel.

*Reference:* Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.

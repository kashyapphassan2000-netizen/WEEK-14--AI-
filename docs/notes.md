# Notes: the IO math behind FlashAttention v1

Day 1 notes, derived by hand. Reference: Dao et al., *FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, sections 3.1 and 3.2.

## 1. Standard attention

For one head with sequence length N and head dim d:

```
S = Q K^T * scale        # (N, N)   scale = 1/sqrt(d)
P = softmax(S, dim=-1)   # (N, N)
O = P V                  # (N, d)
```

The textbook implementation writes S to HBM, reads it back for the softmax, writes P,
then reads P again for `P V`. The N x N matrix makes three round trips through HBM.

At N = 4096 that is 4096 x 4096 x 2 bytes = **32 MB per head** in fp16. An A100 SM has
about 192 KB of SRAM, so the matrix can't stay on-chip.

## 2. Why the bottleneck is memory, not math

A100: about 312 TFLOP/s of fp16 tensor-core compute, but only about 1.5–2 TB/s of HBM bandwidth.
For softmax, the elementwise ops do a few FLOPs per byte loaded, so they are memory-bound.
Standard attention spends most of its time moving S and P, not doing arithmetic.

## 3. Softmax with a running max

Numerically stable softmax for row i:

```
m_i  = max_j S_ij
l_i  = sum_j exp(S_ij - m_i)
P_ij = exp(S_ij - m_i) / l_i
```

## 4. The online-softmax recurrence (the whole trick)

Split the keys into two blocks. Each block, computed on its own, gives:

```
block 1:  m1, l1 = sum exp(S1 - m1),  O1 = sum exp(S1 - m1) V1
block 2:  m2, l2 = sum exp(S2 - m2),  O2 = sum exp(S2 - m2) V2
```

Merge them by rescaling both to a shared max:

```
m  = max(m1, m2)
l  = exp(m1 - m) l1 + exp(m2 - m) l2
O  = exp(m1 - m) O1 + exp(m2 - m) O2
final output = O / l
```

Proof sketch: `exp(S - m1) * exp(m1 - m) = exp(S - m)`, so every term ends up
expressed relative to the same max m, exactly as if we had seen all keys at once.

As a streaming loop over key blocks j, with running state (m, l, acc) per query row:

```
m_new = max(m, rowmax(S_j))
P_j   = exp(S_j - m_new)
alpha = exp(m - m_new)          # how much to shrink the old state
l     = alpha * l + rowsum(P_j)
acc   = alpha * acc + P_j V_j
m     = m_new
```

At the end, O = acc / l and LSE = m + log(l). The LSE is all the backward pass needs
to rebuild P.

## 5. IO complexity

- Standard attention: Θ(N d + N²) HBM accesses. The N² term dominates.
- FlashAttention: Θ(N² d² / M) HBM accesses, where M is SRAM size in elements.

With d = 64 and M ≈ 96K fp16 elements (192 KB), d²/M = 4096/98304 ≈ 1/24. Each query
tile does read every K/V tile, but only tiles of size N·d move per query tile. The
N x N matrix itself never touches HBM.

## 6. The exp2 trick

`exp(x) = exp2(x * log2(e))`. GPUs have a fast native `exp2` instruction. So the kernel
multiplies the scale by log2(e) = 1.4427 once and uses `exp2` everywhere. The saved
LSE is then in base 2. The backward pass uses the same convention, so the two cancel.

## 7. Backward by recomputation

Saved from forward: O, LSE. Given dO:

```
D_i   = rowsum(dO_i * O_i)
P_ij  = exp(S_ij - LSE_i)             # rebuilt from the saved LSE, never stored
dV    = P^T dO
dP    = dO V^T
dS    = P * (dP - D)                  # softmax Jacobian, applied row-wise
dQ    = dS K * scale
dK    = dS^T Q * scale
```

Why `dS = P * (dP - D)`: for one row, `dS_j = P_j (dP_j - sum_k P_k dP_k)`, and
`sum_k P_k dP_k = sum_k P_k (dO · V_k) = dO · O = D`.

## 8. Tile sizes used

| knob | value | why |
|---|---|---|
| BLOCK_M (query rows) | 64 or 128 | autotuned |
| BLOCK_N (key rows) | 32 or 64 | autotuned |
| HEAD_DIM | 16/32/64/128 | compile-time constant, must be a power of 2 |
| num_warps | 4 or 8 | 8 helps at BLOCK_M=128, D=128 |
| num_stages | 2–4 | software pipelining of HBM loads |

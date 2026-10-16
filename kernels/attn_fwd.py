"""Day 4: the FlashAttention v1 forward kernel.

One program owns one tile of BLOCK_M query rows for one (batch, head). It
walks over K and V in tiles of BLOCK_N rows and keeps three running values
per query row in registers:

    m_i  - the largest score seen so far       (for numerical stability)
    l_i  - the softmax denominator so far       (sum of exp(score - m_i))
    acc  - the un-normalized output so far      (sum of exp(score - m_i) * V)

When a new tile raises the max, the old l_i and acc are rescaled by
exp(m_old - m_new) so everything stays in the same units. After the last
tile, acc / l_i is the exact softmax(QK^T)V. The N x N matrix never exists.

Scores are kept in base-2 units (multiplied by log2(e)) so the kernel can use
exp2, which maps to a single hardware instruction. The saved log-sum-exp
(LSE) is in those same base-2 units; the backward pass reads it as-is.
"""
import math

import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg

LOG2E = 1.4426950408889634  # log2(e): exp(x) == exp2(x * LOG2E)


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base, stride_kn, stride_kk, stride_vn, stride_vk,
    offs_m, qk_scale, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Loop over K/V tiles and update (acc, l_i, m_i) with the online softmax."""
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)

    hi = N_CTX
    if CAUSAL:
        # Keys after the last query row in this tile are all masked: skip them.
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        # K tile loaded transposed: [HEAD_DIM, BLOCK_N].
        kT = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                     mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, kT) * qk_scale                       # [BLOCK_M, BLOCK_N]

        valid = mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        # Online softmax update.
        m_ij = tl.maximum(m_i, tl.max(qk, 1))               # new running max
        p = tl.math.exp2(qk - m_ij[:, None])                # [BLOCK_M, BLOCK_N]
        alpha = tl.math.exp2(m_i - m_ij)                    # rescale old stats
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    return acc, l_i, m_i


@triton.autotune(
    configs=autotune_configs(
        [
            cfg(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
            cfg(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3),
            cfg(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=4),
            cfg(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=4),
            cfg(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=3),
        ],
        # Small, unequal blocks in the interpreter so tests hit the
        # multi-tile, padding and causal-diagonal paths at tiny N.
        cfg(BLOCK_M=32, BLOCK_N=16),
    ),
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, LSE, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Grid: (cdiv(N, BLOCK_M), B * H). Writes O and the base-2 LSE per row."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # int64 base offsets: B*H*N*D elements overflows int32 at large shapes.
    q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    # m starts at -inf, so the first tile's alpha = exp2(-inf) = 0 wipes l and acc.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # fold log2(e) in so we can use exp2

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=mask_m[:, None], other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        k_base, v_base, stride_kn, stride_kk, stride_vn, stride_vk,
        offs_m, qk_scale, N_CTX,
        HEAD_DIM, BLOCK_M, BLOCK_N, CAUSAL,
    )

    # Epilogue: normalize once, save LSE for the backward pass.
    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(LSE + off_hz.to(tl.int64) * N_CTX + offs_m, lse, mask=mask_m)
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def _flash_fwd(q, k, v, causal, sm_scale):
    """Launch the forward kernel. Returns (O, LSE in base-2 units, shape [B*H, N])."""
    B, H, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "q, k, v must share a shape"
    assert q.dtype in (torch.float16, torch.bfloat16), "use fp16 or bf16 inputs"
    assert D in (16, 32, 64, 128), "head dim must be 16, 32, 64 or 128"

    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)
    _attn_fwd_kernel[grid](
        q, k, v, sm_scale, lse, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return o, lse


def flash_attn_fwd(q, k, v, causal=False, sm_scale=None):
    """FlashAttention forward (no autograd). Use `flash_attention` for training.

    q, k, v: [B, H, N, D] fp16/bf16. Returns (O, LSE) where LSE has shape
    [B, H, N] and equals torch.logsumexp(Q @ K^T * sm_scale, dim=-1).
    """
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o, lse2 = _flash_fwd(q, k, v, causal, sm_scale)
    return o, (lse2 * math.log(2.0)).view(B, H, N)

"""Day 5: the FlashAttention v1 backward pass, by recomputation.

The forward pass threw away the N x N probability matrix P. Instead of
storing it, the backward pass rebuilds each tile of P on the fly from Q, K
and the saved log-sum-exp (LSE):  P_ij = exp2(S_ij * log2(e) - LSE_i).

With D_i = rowsum(dO_i * O_i), the gradients are:

    dV = P^T @ dO
    dP = dO @ V^T
    dS = P * (dP - D)            # softmax backward
    dQ = dS @ K * scale
    dK = dS^T @ Q * scale

Three kernels:
    _attn_bwd_preprocess  D = rowsum(O * dO), one pass.
    _attn_bwd_dkdv        one program per K/V tile, loops over Q tiles.
    _attn_bwd_dq          one program per Q tile, loops over K/V tiles.

Splitting dK/dV from dQ means every output tile has exactly one writer, so
no atomics are needed. The price is computing QK^T and dO V^T twice.
"""
import math

import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg
from kernels.attn_fwd import _flash_fwd


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Delta[bh, m] = sum_d O[bh, m, d] * dO[bh, m, d]. Inputs are contiguous."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    base = off_hz * N_CTX * HEAD_DIM
    ptrs = base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    o = tl.load(O + ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=mask_m)


_BWD_CONFIGS = autotune_configs(
    [
        cfg(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2),
        cfg(BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2),
        cfg(BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=3),
        cfg(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=3),
        cfg(BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=3),
    ],
    cfg(BLOCK_M=16, BLOCK_N=32),
)


@triton.autotune(configs=_BWD_CONFIGS, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, sm_scale, DO, DK, DV, LSE, Delta,
    stride_z, stride_h, stride_n, stride_d,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Grid: (cdiv(N, BLOCK_N), B * H). Accumulates dK and dV for one K/V tile.

    All of Q, K, V, dO, dK, dV are contiguous [B, H, N, D], so they share strides.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h
    row_base = off_hz.to(tl.int64) * N_CTX

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_CTX

    # K_j and V_j stay in SRAM for the whole program.
    kv_ptrs = base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    k = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    lo = 0
    if CAUSAL:
        # Query rows before this key tile cannot see it: start at its diagonal.
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M

    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX
        q_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
        q = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(LSE + row_base + offs_m, mask=mask_m, other=0.0)
        delta = tl.load(Delta + row_base + offs_m, mask=mask_m, other=0.0)

        # Recompute this tile of P from the saved LSE.
        qk = tl.dot(q, tl.trans(k))                                  # [BLOCK_M, BLOCK_N]
        p = tl.math.exp2(qk * qk_scale - lse[:, None])
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)

        # dV_j += P^T @ dO
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)
        # dP = dO @ V^T ;  dS = P * (dP - D)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        # dK_j += dS^T @ Q   (scale applied once at the end)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)

    dk = dk * sm_scale
    tl.store(DK + kv_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV + kv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


@triton.autotune(configs=_BWD_CONFIGS, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_bwd_dq(
    Q, K, V, sm_scale, DO, DQ, LSE, Delta,
    stride_z, stride_h, stride_n, stride_d,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Grid: (cdiv(N, BLOCK_M), B * H). Accumulates dQ for one Q tile."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h
    row_base = off_hz.to(tl.int64) * N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    q_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    q = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(LSE + row_base + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + row_base + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        kv_ptrs = base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        k = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        p = tl.math.exp2(qk * qk_scale - lse[:, None])
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        # dQ_i += dS @ K
        dq = tl.dot(ds.to(k.dtype), k, dq)

    dq = dq * sm_scale
    tl.store(DQ + q_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


def _flash_bwd(do, q, k, v, o, lse, causal, sm_scale):
    """Launch the three backward kernels. All tensors must be contiguous."""
    B, H, N, D = q.shape
    delta = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    PRE_BLOCK = 64
    _attn_bwd_preprocess[(triton.cdiv(N, PRE_BLOCK), B * H)](
        o, do, delta, N, HEAD_DIM=D, BLOCK_M=PRE_BLOCK,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))

    grid_kv = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * H)
    _attn_bwd_dkdv[grid_kv](
        q, k, v, sm_scale, do, dk, dv, lse, delta,
        *strides, H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    grid_q = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)
    _attn_bwd_dq[grid_q](
        q, k, v, sm_scale, do, dq, lse, delta,
        *strides, H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    """autograd wrapper: forward saves (Q, K, V, O, LSE), never P."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        o, lse = _flash_fwd(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _flash_bwd(do.contiguous(), q, k, v, o, lse, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None


def flash_attention(q, k, v, causal=False, sm_scale=None):
    """Differentiable FlashAttention. q, k, v: [B, H, N, D] fp16/bf16.

    Drop-in for F.scaled_dot_product_attention(q, k, v, is_causal=causal).
    """
    return FlashAttnFunc.apply(q, k, v, causal, sm_scale)

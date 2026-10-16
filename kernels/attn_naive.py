"""Day 3: naive ("standard") attention in Triton. This is the baseline.

It does exactly what the textbook formula says and what eager PyTorch does:

    S = Q @ K^T * scale      -> written to HBM   (N x N per head)
    P = softmax(S)           -> read + written   (N x N per head)
    O = P @ V                -> read again

Three kernels, three round trips through HBM for the N x N matrix. That
O(N^2) memory traffic is the cost FlashAttention removes, so this is the
baseline the "speedup vs naive" number is measured against.

Note: the guide's single-kernel version keeps all of K and V for one head in
on-chip memory. At N=4096, D=64 that is 512 KB per tensor, which does not fit
in shared memory, so it fails to compile at the sizes we benchmark. Writing S
to HBM is the honest "standard attention" baseline from the paper.
"""
import torch
import triton
import triton.language as tl

from kernels._utils import is_interpreter


@triton.jit
def _qk_kernel(
    Q, K, S, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """S[b, h, m, n] = scale * sum_d Q[b, h, m, d] * K[b, h, n, d] for one tile."""
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    s_base = S + off_hz.to(tl.int64) * N_CTX * N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX
    mask_n = offs_n < N_CTX

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=mask_m[:, None], other=0.0)
    # Load K already transposed: [HEAD_DIM, BLOCK_N].
    kT = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                 mask=mask_n[None, :], other=0.0)
    s = tl.dot(q, kT) * sm_scale

    s_ptrs = s_base + offs_m[:, None] * N_CTX + offs_n[None, :]
    tl.store(s_ptrs, s.to(S.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _softmax_kernel(S, P, N_CTX, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    """Row-wise softmax of one row of S, with the max subtracted for stability."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    valid = offs < N_CTX
    if CAUSAL:
        valid = valid & (offs <= row % N_CTX)

    s = tl.load(S + row * N_CTX + offs, mask=valid, other=float("-inf")).to(tl.float32)
    s = s - tl.max(s, axis=0)
    p = tl.exp(s)
    p = p / tl.sum(p, axis=0)
    tl.store(P + row * N_CTX + offs, p.to(P.dtype.element_ty), mask=offs < N_CTX)


@triton.jit
def _pv_kernel(
    P, V, Out,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """O[b, h, m, :] = sum_n P[b, h, m, n] * V[b, h, n, :] for one row tile."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    p_base = P + off_hz.to(tl.int64) * N_CTX * N_CTX
    v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        p = tl.load(p_base + offs_m[:, None] * N_CTX + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def naive_attention(q, k, v, causal=False, sm_scale=None):
    """Standard attention that materializes the N x N score matrix in HBM.

    q, k, v: [B, H, N, D] fp16/bf16 tensors. Returns O with the same shape.
    """
    B, H, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "q, k, v must share a shape"
    assert D in (16, 32, 64, 128), "head dim must be 16, 32, 64 or 128"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # The N x N intermediate. Kept in the input dtype, like eager PyTorch.
    s = torch.empty((B * H, N, N), device=q.device, dtype=q.dtype)
    o = torch.empty_like(q)

    BLOCK_M, BLOCK_N = (32, 16) if is_interpreter() else (64, 64)

    grid_qk = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N), B * H)
    _qk_kernel[grid_qk](
        q, k, s, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    # Softmax in place: S becomes P. One program per row.
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    _softmax_kernel[(B * H * N,)](s, s, N, CAUSAL=causal, BLOCK=BLOCK, num_warps=num_warps)

    grid_pv = (triton.cdiv(N, BLOCK_M), B * H)
    _pv_kernel[grid_pv](
        s, v, o,
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o

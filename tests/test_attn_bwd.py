import pytest
import torch
import torch.nn.functional as F

from helpers import BF16_OK, ON_GPU, rand_qkv
from kernels.attn_bwd import flash_attention

SHAPES = (
    [(1, 4, 256, 64), (2, 4, 512, 64), (1, 4, 1000, 64), (1, 2, 512, 128)]
    if ON_GPU
    else [(1, 1, 64, 32), (1, 2, 100, 64), (1, 1, 50, 16)]
)


def _grads(fn, q, k, v, do):
    q, k, v = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    o = fn(q, k, v)
    o.backward(do)
    return o, q.grad, k.grad, v.grad


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_bwd_matches_torch(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    do = torch.randn_like(q)

    o, dq, dk, dv = _grads(lambda a, b, c: flash_attention(a, b, c, causal=causal), q, k, v, do)
    # Reference in fp32, so we compare against the true gradient, not fp16 noise.
    ref = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float(), is_causal=causal),
        q, k, v, do.float(),
    )

    torch.testing.assert_close(o, ref[0].to(o.dtype), rtol=1e-2, atol=1e-2)
    for name, got, want in zip("qkv", (dq, dk, dv), ref[1:]):
        torch.testing.assert_close(got, want.to(got.dtype), rtol=2e-2, atol=2e-2,
                                   msg=lambda m, n=name: f"d{n}: {m}")


def test_bwd_sum_loss_as_in_guide():
    """The guide's exact check: o.sum().backward() at N=256, D=64 (smaller on CPU)."""
    B, H, N, D = (1, 4, 256, 64) if ON_GPU else (1, 1, 64, 64)
    q, k, v = rand_qkv(B, H, N, D, requires_grad=True)
    qr, kr, vr = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(qr, kr, vr).sum().backward()
    flash_attention(q, k, v).sum().backward()
    for got, want in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not BF16_OK, reason="needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16")
@pytest.mark.parametrize("causal", [False, True])
def test_bwd_bf16(causal):
    q, k, v = rand_qkv(1, 4, 512, 64, dtype=torch.bfloat16)
    do = torch.randn_like(q)
    _, *got = _grads(lambda a, b, c: flash_attention(a, b, c, causal=causal), q, k, v, do)
    _, *want = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float(), is_causal=causal),
        q, k, v, do.float(),
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w.to(g.dtype), rtol=5e-2, atol=5e-2)

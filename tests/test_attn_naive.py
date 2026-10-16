import pytest
import torch

from helpers import ON_GPU, rand_qkv, reference_attention
from kernels.attn_naive import naive_attention

SHAPES = (
    [(1, 4, 256, 64), (2, 8, 512, 64), (1, 4, 1000, 128)]
    if ON_GPU
    else [(1, 2, 64, 32), (2, 2, 100, 64), (1, 1, 77, 16)]
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_naive_vs_torch(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    o = naive_attention(q, k, v, causal=causal)
    torch_o = reference_attention(q, k, v, causal=causal)
    torch.testing.assert_close(o, torch_o, rtol=1e-2, atol=1e-2)

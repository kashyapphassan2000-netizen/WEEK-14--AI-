import pytest
import torch

from helpers import BF16_OK, DEVICE, ON_GPU, rand_qkv, reference_attention
from kernels.attn_fwd import flash_attn_fwd

SHAPES = (
    [(1, 4, 512, 64), (2, 8, 1024, 64), (1, 16, 2048, 64), (1, 8, 1024, 128), (1, 4, 1000, 64)]
    if ON_GPU
    else [(1, 2, 64, 32), (2, 2, 100, 64), (1, 1, 77, 16), (1, 1, 48, 128)]
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_flash_fwd(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    o, _ = flash_attn_fwd(q, k, v, causal=causal)
    torch.testing.assert_close(o, reference_attention(q, k, v, causal), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_fwd_lse(causal):
    """Debug-playbook checkpoint 2: m + log(l) must equal torch.logsumexp(S)."""
    B, H, N, D = (1, 2, 256, 64) if ON_GPU else (1, 1, 70, 32)
    q, k, v = rand_qkv(B, H, N, D)
    _, lse = flash_attn_fwd(q, k, v, causal=causal)
    s = (q.float() @ k.float().transpose(-1, -2)) / D ** 0.5
    if causal:
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, dim=-1), rtol=1e-3, atol=1e-3)


def test_flash_fwd_strided():
    """q/k/v as non-contiguous views of a [B, N, H, D] tensor (common in real models)."""
    B, H, N, D = (2, 4, 512, 64) if ON_GPU else (1, 2, 64, 32)
    torch.manual_seed(0)
    q, k, v = [torch.randn(B, N, H, D, device=DEVICE, dtype=torch.float16).transpose(1, 2)
               for _ in range(3)]
    o, _ = flash_attn_fwd(q, k, v)
    torch.testing.assert_close(o, reference_attention(q, k, v), rtol=1e-2, atol=1e-2)


# Triton's CPU interpreter computes tl.dot on bf16 inputs incorrectly (it is a
# numpy emulation with no bf16 type), and pre-Ampere GPUs have no bf16 tensor
# cores, so this only runs on an Ampere-or-newer GPU.
@pytest.mark.skipif(not BF16_OK, reason="needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16")
@pytest.mark.parametrize("causal", [False, True])
def test_flash_fwd_bf16(causal):
    q, k, v = rand_qkv(2, 4, 512, 64, dtype=torch.bfloat16)
    o, _ = flash_attn_fwd(q, k, v, causal=causal)
    torch.testing.assert_close(o, reference_attention(q, k, v, causal), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not ON_GPU, reason="PASS gate size; needs a GPU")
def test_pass_gate_forward_n2048():
    """Appendix C, criterion 1: N=2048, D=64, fp16, rtol=atol=1e-2."""
    q, k, v = rand_qkv(1, 16, 2048, 64)
    o, _ = flash_attn_fwd(q, k, v)
    torch.testing.assert_close(o, reference_attention(q, k, v), rtol=1e-2, atol=1e-2)

import torch
import torch.nn.functional as F

from conftest import DEVICE, ON_GPU  # noqa: F401


def rand_qkv(B, H, N, D, dtype=torch.float16, requires_grad=False):
    torch.manual_seed(0)
    return [
        torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=requires_grad)
        for _ in range(3)
    ]


def reference_attention(q, k, v, causal=False):
    """fp32 SDPA reference, cast back to the input dtype."""
    o = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    return o.to(q.dtype)

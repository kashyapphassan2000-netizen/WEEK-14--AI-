import pytest
import torch

from helpers import DEVICE, ON_GPU
from kernels.matmul import matmul

SHAPES = (
    [(512, 512, 512), (1024, 1024, 1024), (768, 1024, 512)]
    if ON_GPU
    else [(64, 64, 64), (100, 72, 48), (37, 129, 65)]
)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matmul_correctness(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(matmul(a, b), ref, rtol=1e-2, atol=1e-2)


def test_matmul_non_contiguous():
    torch.manual_seed(0)
    a = torch.randn((64, 96), device=DEVICE, dtype=torch.float16)
    b = torch.randn((80, 96), device=DEVICE, dtype=torch.float16).t()  # strided view
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(matmul(a, b), ref, rtol=1e-2, atol=1e-2)

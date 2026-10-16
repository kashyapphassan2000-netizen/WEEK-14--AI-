"""Day 2: tiled fp16 matmul in Triton.

The "hello world" kernel. It exercises everything the attention kernels need:
the program-id grid, pointer arithmetic with strides, masked `tl.load` /
`tl.store`, and `tl.dot` with an fp32 accumulator.
"""
import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg


@triton.autotune(
    configs=autotune_configs(
        [
            cfg(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
            cfg(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
            cfg(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
            cfg(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
        ],
        cfg(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=4),
    ),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Each program computes one [BLOCK_M, BLOCK_N] tile of C = A @ B."""
    # Group-ordered program ids: neighbouring programs share rows of A,
    # which improves L2 cache reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Wrap-around (% M, % N) keeps loads in bounds; the store mask discards them.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B for 2-D fp16/bf16 tensors. Accumulates in fp32."""
    assert a.ndim == 2 and b.ndim == 2, "matmul expects 2-D tensors"
    assert a.shape[1] == b.shape[0], "K mismatch"
    assert a.dtype == b.dtype, "dtype mismatch"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":  # python -m kernels.matmul
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(512, 512, device=dev, dtype=torch.float16)
    b = torch.randn(512, 512, device=dev, dtype=torch.float16)
    diff = (matmul(a, b).float() - a.float() @ b.float()).abs().max().item()
    print(f"max abs diff vs torch: {diff:.4e}")

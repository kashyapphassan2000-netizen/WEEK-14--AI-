"""Pick where the kernels run before any kernel module is imported.

- CUDA GPU present  -> real kernels, the guide's full test sizes.
- No GPU            -> Triton's CPU interpreter (TRITON_INTERPRET=1), small sizes.
  Same kernel code, same math, ~1000x slower, so shapes are shrunk.
"""
import os

import torch

ON_GPU = torch.cuda.is_available()
if not ON_GPU:
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if ON_GPU else "cpu"

# bf16 tensor cores exist from Ampere (compute capability 8.x: A100, RTX 30xx/40xx) on.
# A T4 (7.5) or V100 (7.0) has none, so bf16 tests only run on 8.x+.
BF16_OK = ON_GPU and torch.cuda.get_device_capability()[0] >= 8

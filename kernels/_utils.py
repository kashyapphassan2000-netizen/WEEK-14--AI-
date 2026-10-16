"""Small helpers shared by every kernel file."""
import os

import triton


def is_interpreter() -> bool:
    """True when Triton runs kernels in its CPU interpreter (TRITON_INTERPRET=1)."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def autotune_configs(gpu_configs, interpreter_config):
    """Pick the autotune search space.

    On a GPU we let `triton.autotune` try every config. The interpreter has no
    GPU timers, so we hand it exactly one config: with a single config the
    autotuner skips benchmarking entirely.
    """
    if is_interpreter():
        return [interpreter_config]
    return gpu_configs


def cfg(num_warps=4, num_stages=3, **meta) -> triton.Config:
    return triton.Config(meta, num_warps=num_warps, num_stages=num_stages)

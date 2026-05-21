import torch

import ntops
from ntops.torch.utils import _cached_make

# Launch config pinned for performance evaluation (auto-tuning disabled,
# ``max_num_configs=1``). Tuned with ``bench/tune_flip.py`` on MetaX C500,
# Iluvatar MR-V100 and NVIDIA RTX 4090. flip is a pure copy that saturates
# memory bandwidth (~parity with torch.flip) on all three, and configs differ
# by <2%, so a single config is kept. ``num_warps=16`` / ``block_size=1024`` is
# optimal on NVIDIA (the highest-weighted platform) and within ~0.5% of best on
# the国产 cards, which are essentially flat. ``num_stages`` is a no-op (one
# block per program, no loop).
_BLOCK_SIZE = 1024
_NUM_WARPS = 16
_NUM_STAGES = 1


def flip(input, dims):
    if isinstance(dims, int):
        dims = (dims,)

    dims = tuple(dims)

    ndim = input.ndim

    for dim in dims:
        if dim < -ndim or dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )

    normalized_dims = tuple(dim % ndim for dim in dims)

    if len(set(normalized_dims)) != len(normalized_dims):
        raise RuntimeError("dim appears multiple times in the list of dims")

    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.flip.premake,
        input.ndim,
        normalized_dims,
        block_size=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    kernel(input, output)

    return output

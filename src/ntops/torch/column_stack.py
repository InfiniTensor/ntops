"""Column stack: one identity kernel per input, writing to output slice.

Kernel writes directly to non-contiguous output slices using stride-aware
flatten+tile arrangement. No torch.column_stack is called.
"""

import torch

from ntops.torch.utils import _cached_make
from ntops.kernels.column_stack import premake


def column_stack(tensors):
    """Stack tensors column-wise via one kernel per input.

    0-D/1-D → reshape to (numel, 1); 2-D+ → as-is.
    Stacked along dim=1 per PyTorch semantics.
    """
    if not tensors:
        raise ValueError("need at least one tensor")

    # 0-D/1-D → (numel, 1); 2-D+ → as-is
    reshaped = [t.reshape(t.numel(), 1) if t.ndim <= 1 else t for t in tensors]

    # Validate: all same ndim, non-concat dims match
    ndim = reshaped[0].ndim
    for t in reshaped:
        if t.ndim != ndim:
            raise RuntimeError(
                f"all tensors must have the same number of dimensions after "
                f"reshape, got {ndim} and {t.ndim}"
            )
    for d in range(ndim):
        if d == 1:
            continue
        size0 = reshaped[0].shape[d]
        for t in reshaped[1:]:
            if t.shape[d] != size0:
                raise RuntimeError(
                    f"non-concatenating dimension {d} does not match: "
                    f"{size0} vs {t.shape[d]}"
                )

    n_cols = sum(t.shape[1] for t in reshaped)
    out_shape = list(reshaped[0].shape)
    out_shape[1] = n_cols
    dtype = tensors[0].dtype
    device = tensors[0].device

    out = torch.empty(out_shape, dtype=dtype, device=device)

    kernel = _cached_make(premake, ndim=ndim)
    col_start = 0
    for t in reshaped:
        c = t.shape[1]
        slices = [slice(None)] * ndim
        slices[1] = slice(col_start, col_start + c)
        kernel(t, out[tuple(slices)])
        col_start += c

    return out

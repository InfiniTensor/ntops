"""Cartesian product: one kernel per input column.

Each kernel computes one column of the cartesian product using:
    output[r] = input[(r // repeat_after) % L]
where repeat_after = product(sizes[col+1:]).
"""

import torch

from ntops.torch.utils import _cached_make
from ntops.kernels.cartesian_prod import premake


def cartesian_prod(*tensors):
    """Cartesian product of 1D tensors.

    For N input tensors with sizes [L0, L1, ..., L_{N-1}], output is
    (total_rows, N) where total_rows = product of all sizes.
    Each column col cycles through input[col] every repeat_after[col] rows.

    No torch.cartesian_prod or torch.meshgrid is called.
    """
    if not tensors:
        raise ValueError("cartesian_prod requires at least one tensor")

    # Validate all inputs are 1D
    for i, t in enumerate(tensors):
        if t.ndim != 1:
            raise RuntimeError(
                f"cartesian_prod expected 1D tensors, "
                f"but got {t.ndim}D at position {i}"
            )

    # Single input: return as-is (PyTorch semantic)
    if len(tensors) == 1:
        return tensors[0]

    sizes = [t.shape[0] for t in tensors]
    n_cols = len(sizes)
    total_rows = 1
    for s in sizes:
        total_rows *= s

    dtype = tensors[0].dtype
    device = tensors[0].device

    # Allocate output (handles zero-size gracefully)
    output = torch.empty(total_rows, n_cols, dtype=dtype, device=device)

    if total_rows == 0:
        return output  # one or more inputs have length 0

    # Compute repeat_after for each column:
    # repeat_after[col] = product(sizes[col+1:]) (1 for last column)
    repeat_after = [1] * n_cols
    for col in range(n_cols - 2, -1, -1):
        repeat_after[col] = repeat_after[col + 1] * sizes[col + 1]

    for col, t in enumerate(tensors):
        if sizes[col] == 0:
            continue
        kernel = _cached_make(
            premake, repeat_after=repeat_after[col], size=sizes[col]
        )
        kernel(t, output[:, col], repeat_after[col], sizes[col])

    return output

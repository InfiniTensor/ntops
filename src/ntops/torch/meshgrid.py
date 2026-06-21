"""Meshgrid: reuses cartesian_prod kernel for each grid axis.

For output grid k with shape (S0, ..., SN-1):
    output_k[flat_idx] = input_k[unravel(flat_idx)[k]]
                      = input_k[(flat_idx // repeat_after[k]) % Sk]

This is identical to the cartesian_prod column formula.
No torch.meshgrid is called.
"""

import torch

from ntops.torch.utils import _cached_make
from ntops.kernels.cartesian_prod import premake


def meshgrid(*tensors, indexing="ij"):
    """Generate ND coordinate grids from 1D tensors.

    Uses the cartesian_prod kernel: for each input, computes the flat
    index mapping and reshapes the result to the full grid shape.

    Args:
        *tensors: 1-D tensors.
        indexing: 'ij' (matrix convention) or 'xy' (Cartesian convention).
    """
    if indexing not in ("ij", "xy"):
        raise ValueError(
            f"indexing must be 'ij' or 'xy', got '{indexing}'"
        )

    # Support single list/tuple argument: torch.meshgrid([x, y])
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if not tensors:
        return tuple()

    # 0-D tensors → 1-D of length 1
    tensors = tuple(t.reshape(1) if t.ndim == 0 else t for t in tensors)

    # For 'xy' indexing, swap the first two tensors' roles
    if indexing == "xy" and len(tensors) >= 2:
        tensors = (tensors[1], tensors[0], *tensors[2:])

    sizes = [t.shape[0] for t in tensors]
    output_shape = tuple(sizes)
    dtype = tensors[0].dtype
    device = tensors[0].device

    total_el = 1
    for s in sizes:
        total_el *= s

    if total_el == 0:
        # One or more inputs have length 0 → all outputs are empty
        return tuple(
            torch.empty(output_shape, dtype=dtype, device=device)
            for _ in tensors
        )

    # Compute repeat_after for each grid axis
    n = len(sizes)
    repeat_after = [1] * n
    for k in range(n - 2, -1, -1):
        repeat_after[k] = repeat_after[k + 1] * sizes[k + 1]

    outs = []
    for k, t in enumerate(tensors):
        flat_out = torch.empty(total_el, dtype=dtype, device=device)
        kernel = _cached_make(
            premake, repeat_after=repeat_after[k], size=sizes[k]
        )
        kernel(t, flat_out, repeat_after[k], sizes[k])
        outs.append(flat_out.reshape(output_shape))

    # For 'xy' indexing, swap the first two outputs back
    if indexing == "xy" and len(outs) >= 2:
        outs[0], outs[1] = outs[1], outs[0]

    return tuple(outs)

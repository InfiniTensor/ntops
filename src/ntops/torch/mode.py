"""Mode torch wrapper: permute reduction dim to last, call kernel, restore shape.

Pads reduction dim to power-of-2 with zero-fill; kernel masks padding lanes.
No torch.mode is called.
"""

import torch
import torch.nn.functional as F

from ntops.torch.utils import _cached_make
from ntops.kernels.mode import premake


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def mode(input, dim=-1, keepdim=False):
    """Return the mode (most frequent value) along a dimension.

    Returns (values, indices) tuple.
    Tie-breaking is deterministic but may differ from torch.mode on CUDA
    for exact ties; see REFERENCE.md.
    """
    ndim = input.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected [-{ndim}, {ndim-1}], got {dim})"
        )

    K_orig = input.shape[dim]
    if K_orig == 0:
        raise ValueError("mode reduction dimension cannot be empty")
    K_tile = _next_power_of_2(K_orig)

    # Permute so reduction dim is last
    perm = [i for i in range(ndim) if i != dim] + [dim]
    permuted = input.permute(perm)  # (..., K_orig)
    permuted_shape = list(permuted.shape)

    if K_orig != K_tile:
        # Zero-pad to power-of-2; kernel masks padding lanes
        pad = (0, K_tile - K_orig)
        permuted = F.pad(permuted, pad)
        permuted_shape[-1] = K_tile

    num_rows = 1
    for s in permuted_shape[:-1]:
        num_rows *= s

    flat_input = permuted.reshape(num_rows, K_tile)

    values_2d = torch.empty(num_rows, 1, dtype=input.dtype, device=input.device)
    indices_2d = torch.empty(num_rows, 1, dtype=torch.int64, device=input.device)

    kernel = _cached_make(premake, K_orig=K_orig, K_tile=K_tile)
    kernel(flat_input, values_2d, indices_2d, K_orig, K_tile)

    out_shape = tuple(permuted_shape[:-1])
    values = values_2d.reshape(out_shape)
    indices = indices_2d.reshape(out_shape)

    if keepdim:
        out_shape_kd = list(out_shape)
        out_shape_kd.insert(dim, 1)
        values = values.reshape(tuple(out_shape_kd))
        indices = indices.reshape(tuple(out_shape_kd))

    return values, indices

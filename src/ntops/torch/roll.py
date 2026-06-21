"""Roll torch wrapper: permute + single-kernel roll per dimension.

Strategy:
- dims=None: flatten → 1D roll with ndim=1 kernel → reshape
- dims specified: for each (shift, dim), permute target dim to position 0,
  call kernel, then permute back.

All data movement goes through compiled ntops kernels (ntl.load with
manual pointer arithmetic). No torch.roll is called.
"""

import torch

from ntops.torch.utils import _cached_make
from ntops.kernels.roll import premake


def roll(input, shifts, dims=None):
    """Roll the tensor along the given dimension(s).

    Args:
        input: input tensor.
        shifts: int or tuple of ints — amount to shift.
        dims: int or tuple of ints, or None to flatten and roll.
    """
    if dims is None:
        # Flatten → 1-D roll → reshape
        shape = input.shape
        flat = input.flatten()
        if flat.shape[0] == 0:
            return flat.clone().reshape(shape)
        shift = int(shifts) % flat.shape[0]
        if shift == 0:
            return flat.clone().reshape(shape)
        out_flat = torch.empty_like(flat)
        kernel = _cached_make(premake, ndim=1, shift=shift)
        kernel(flat, out_flat, shift)
        return out_flat.reshape(shape)

    # Normalize shifts and dims to tuples
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)

    if len(shifts) != len(dims):
        raise ValueError(
            f"shifts and dims must have the same length, "
            f"got {len(shifts)} and {len(dims)}"
        )

    # Handle negative dims and validate
    _dims = []
    for d in dims:
        if d < -input.ndim or d >= input.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-input.ndim}, {input.ndim - 1}], but got {d})"
            )
        _dims.append(d if d >= 0 else input.ndim + d)
    dims = tuple(_dims)

    # Build per-dim shift array (accumulate shifts for repeated dims)
    shift_per_dim = [0] * input.ndim
    for s, d in zip(shifts, dims):
        shift_per_dim[d] = (shift_per_dim[d] + s) % max(input.shape[d], 1)

    # Roll each dimension independently
    result = input
    for d, s in enumerate(shift_per_dim):
        if s == 0:
            continue
        result = _roll_along_dim(result, s, d)

    return result


def _roll_along_dim(tensor, shift, dim):
    """Roll tensor along a single dimension.

    Permutes the target dim to position 0, calls the roll kernel,
    then permutes back. Permutations are stride-only views (no copies).
    """
    ndim = tensor.ndim

    # Create permutation that puts `dim` at position 0
    perm = [dim] + [i for i in range(ndim) if i != dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    # Permute: target dim → dim 0 (view only, no copy)
    permuted = tensor.permute(perm)
    out_permuted = torch.empty_like(permuted)

    # Apply shift, made positive (kernel handles subtract of positive shift)
    size = permuted.shape[0]
    shift_mod = shift % size

    kernel = _cached_make(premake, ndim=ndim, shift=shift_mod)
    kernel(permuted, out_permuted, shift_mod)

    # Permute back
    return out_permuted.permute(inv_perm)

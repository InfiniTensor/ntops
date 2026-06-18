"""Kernel-based mode computation.

Uses O(n²) per-tile frequency counting via DSL. Out-of-bounds
elements (loaded as NaN) are skipped via x != x detection.
"""
import torch

import ntops
from ntops.torch.utils import _cached_make


def mode(input, dim=-1, keepdim=False):
    if dim < 0:
        dim += input.ndim

    if input.ndim == 1:
        input_2d = input.view(1, -1)
        kernel_output = torch.zeros(1, dtype=input.dtype, device=input.device)
        kernel = _cached_make(ntops.kernels.mode.premake, input_2d.ndim)
        kernel(input_2d, kernel_output)
        values = kernel_output.squeeze(0)

        with torch.no_grad():
            mask = input == values
            idx_tensor = torch.arange(input.shape[0], device=input.device)
            indices = (idx_tensor * mask).max(dim=0).values

        if keepdim:
            values = values.unsqueeze(0)

        return values, indices

    # Multi-dim: move target dim to last (permute keeps non-target order)
    dims = list(range(input.ndim))
    dims.remove(dim)
    dims.append(dim)
    input_permuted = input.permute(*dims)

    # Flatten to 2D: (groups, dim_size)
    input_2d = input_permuted.reshape(-1, input_permuted.shape[-1])

    kernel_output = torch.empty(input_2d.shape[0], dtype=input.dtype, device=input.device)
    kernel = _cached_make(ntops.kernels.mode.premake, input_2d.ndim)
    kernel(input_2d, kernel_output)

    # Reshape: one mode per group, preserving non-target dim order
    values = kernel_output.reshape(input_permuted.shape[:-1])

    # Indices: find LAST occurrence of mode value (matching torch.mode)
    with torch.no_grad():
        idx_tensor = torch.arange(input.shape[dim], device=input.device, dtype=torch.long)
        idx_shape = [1] * input.ndim
        idx_shape[dim] = input.shape[dim]
        idx_broadcast = idx_tensor.view(idx_shape)

        mask = input == values.unsqueeze(dim)
        indices = (idx_broadcast * mask).max(dim=dim, keepdim=True).values

    if keepdim:
        values = values.unsqueeze(dim)
    else:
        indices = indices.squeeze(dim)

    return values, indices

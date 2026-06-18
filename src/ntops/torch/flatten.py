import torch

import ntops
from ntops.torch.utils import _cached_make


def flatten(input, start_dim=0, end_dim=-1):
    if end_dim < 0:
        end_dim = input.ndim + end_dim

    if start_dim < 0:
        start_dim = input.ndim + start_dim

    flattened_numel = 1

    for dim in range(start_dim, end_dim + 1):
        flattened_numel *= input.shape[dim]

    out_shape = input.shape[:start_dim] + (flattened_numel,) + input.shape[end_dim + 1 :]

    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Reshape input to match output ndim so the kernel can process both uniformly.
    reshaped_input = input.reshape(out_shape)

    kernel = _cached_make(ntops.kernels.flatten.premake, out.ndim)

    kernel(reshaped_input, out)

    return out

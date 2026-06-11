import torch

import ntops
from ntops.torch.utils import _cached_make


def roll(input, shifts, dims=None):
    if input.ndim == 1:
        return _roll_1d(input, shifts, 0 if dims is None else dims)

    return torch.roll(input, shifts, dims)


def _roll_1d(input, shift, dim):
    """Roll a 1D tensor using ninetoothed kernel."""
    n = input.shape[0]
    shift = shift % n

    input_2d = input.view(1, n)
    output_2d = torch.empty(1, n, device=input.device, dtype=input.dtype)
    size_t = torch.tensor([n], dtype=torch.float32, device=input.device)
    shift_t = torch.tensor([shift], dtype=torch.float32, device=input.device)

    kernel = _cached_make(ntops.kernels.roll.premake, input_2d.ndim)
    kernel(input_2d, size_t, shift_t, output_2d)

    return output_2d.view(n)

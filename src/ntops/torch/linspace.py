import torch

import ntops
from ntops.torch.utils import _cached_make


def linspace(start, end, steps, *, dtype=None, device=None):
    """
    Create a 1D tensor of evenly spaced values from start to end.

    Args:
        start: Starting value
        end: Ending value
        steps: Number of points
        dtype: Data type of the output tensor (defaults to float32)
        device: Device to place the output on

    Returns:
        A 1D tensor of shape (steps,) with evenly spaced values
    """
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(steps, int):
        raise TypeError(f"steps must be an integer, got {type(steps)}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    # Special case: single element
    if steps == 1:
        return torch.tensor([start], dtype=dtype, device=device)

    # Precompute step value
    step_val = (end - start) / (steps - 1)

    output = torch.empty(steps, dtype=dtype, device=device)

    kernel = _cached_make(ntops.kernels.linspace.premake, 1)
    kernel(output, start, step_val)

    return output

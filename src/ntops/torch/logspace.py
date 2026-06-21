import torch

import ntops
from ntops.torch.utils import _cached_make


def logspace(start, end, steps, base=10.0, *, dtype=None, device=None):
    """
    Create a 1D tensor of values evenly spaced on a log scale.

    The values are base^start, base^(start + step), ..., base^end, where
    step = (end - start) / (steps - 1).

    Uses a single fused GPU kernel (linspace + pow) for efficiency.

    Args:
        start: Starting exponent value
        end: Ending exponent value
        steps: Number of points
        base: Base of the log space (default: 10.0)
        dtype: Data type of the output tensor (defaults to float32)
        device: Device to place the output on

    Returns:
        A 1D tensor of shape (steps,) with logarithmically spaced values

    Examples:
        >>> logspace(0, 2, 3, base=10)
        tensor([1., 10., 100.])
        >>> logspace(0, 1, 4, base=2)
        tensor([1., 1.2599, 1.5874, 2.])
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
        return torch.tensor([base ** start], dtype=dtype, device=device)

    # Precompute step value for the exponent
    step_val = (end - start) / (steps - 1)

    output = torch.empty(steps, dtype=dtype, device=device)

    kernel = _cached_make(ntops.kernels.logspace.premake, 1)
    kernel(output, start, step_val, base)

    return output

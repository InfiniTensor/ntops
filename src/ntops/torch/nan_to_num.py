import torch

import ntops
from ntops.torch.utils import _cached_make


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN, positive infinity, and negative infinity values in a tensor.

    Args:
        x: Input tensor
        nan: Value to replace NaN with (default: 0.0)
        posinf: Value to replace positive infinity with.
                If None, uses the max value for the tensor's dtype.
        neginf: Value to replace negative infinity with.
                If None, uses the min value for the tensor's dtype.

    Returns:
        A tensor with NaN and infinity values replaced

    Examples:
        >>> x = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.0])
        >>> nan_to_num(x)
        tensor([0.0000e+00, 3.4028e+38, -3.4028e+38, 1.0000e+00])
    """
    # Integer types cannot represent NaN or Inf — return clone
    if not x.dtype.is_floating_point:
        return x.clone()

    # 0-dim scalar tensors — handle directly without kernel
    if x.ndim == 0:
        if torch.isnan(x):
            return torch.tensor(nan, dtype=x.dtype, device=x.device)
        if torch.isposinf(x):
            if posinf is None:
                posinf = torch.finfo(x.dtype).max
            return torch.tensor(posinf, dtype=x.dtype, device=x.device)
        if torch.isneginf(x):
            if neginf is None:
                neginf = torch.finfo(x.dtype).min
            return torch.tensor(neginf, dtype=x.dtype, device=x.device)
        return x.clone()

    if posinf is None:
        posinf = torch.finfo(x.dtype).max
    if neginf is None:
        neginf = torch.finfo(x.dtype).min

    # Broadcast replacement values to match input shape for kernel compatibility
    nan_val = torch.full_like(x, nan)
    posinf_val = torch.full_like(x, posinf)
    neginf_val = torch.full_like(x, neginf)

    output = torch.empty_like(x)

    kernel = _cached_make(ntops.kernels.nan_to_num.premake, x.ndim)
    kernel(x, nan_val, posinf_val, neginf_val, output)

    return output

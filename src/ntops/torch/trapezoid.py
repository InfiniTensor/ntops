import torch

import ntops
from ntops.torch.utils import _cached_make


def trapezoid(y, x=None, *, dx=1.0, dim=-1):
    # Move the integration dimension to the last axis for uniform handling
    if dim != -1 and dim != y.ndim - 1:
        y = y.transpose(dim, -1)
        if x is not None and x.ndim == y.ndim:
            x = x.transpose(dim, -1)

    if x is None:
        mid = (y[..., 1:] + y[..., :-1]) * 0.5
        areas = mid * dx
    else:
        if x.ndim == 1:
            dx_vals = x[1:] - x[:-1]
        else:
            dx_vals = x[..., 1:] - x[..., :-1]
        mid = (y[..., 1:] + y[..., :-1]) * 0.5
        areas = mid * dx_vals

    output = torch.empty_like(areas)
    kernel = _cached_make(ntops.kernels.trapezoid.premake, areas.ndim)
    kernel(areas, output)

    result = output.sum(dim=-1)

    return result

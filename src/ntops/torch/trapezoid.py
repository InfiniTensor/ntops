import torch

import ntops
from ntops.torch.utils import _cached_make


def trapezoid(y, x=None, *, dx=1.0, dim=-1):
    if dim != -1 and dim != y.ndim - 1:
        y = y.transpose(dim, -1)
        if x is not None and x.ndim == y.ndim:
            x = x.transpose(dim, -1)

    if x is None:
        areas = (y[..., 1:] + y[..., :-1]) * 0.5 * dx
    else:
        if x.ndim == 1:
            dx_vals = x[1:] - x[:-1]
        else:
            dx_vals = x[..., 1:] - x[..., :-1]
        areas = (y[..., 1:] + y[..., :-1]) * 0.5 * dx_vals

    output = torch.zeros(*areas.shape[:-1], 1, dtype=areas.dtype, device=areas.device)

    kernel = _cached_make(ntops.kernels.trapezoid.premake, areas.ndim)
    kernel(areas, output)

    return output.squeeze(-1)

import torch

import ntops
from ntops.torch.utils import _cached_make


def meshgrid(*tensors, indexing="ij"):
    if len(tensors) > 2:
        return torch.meshgrid(*tensors, indexing=indexing)

    x, y = tensors[0], tensors[1]

    if indexing == "ij":
        nx, ny = x.size(0), y.size(0)
        x_grid = x.view(-1, 1).expand(nx, ny)
        y_grid = y.view(1, -1).expand(nx, ny)
    else:
        ny, nx = y.size(0), x.size(0)
        x_grid = x.view(1, -1).expand(ny, nx)
        y_grid = y.view(-1, 1).expand(ny, nx)

    X = torch.empty_like(x_grid)
    Y = torch.empty_like(y_grid)

    kernel = _cached_make(ntops.kernels.meshgrid.premake, X.ndim)
    kernel(x_grid, y_grid, X, Y)

    return X, Y

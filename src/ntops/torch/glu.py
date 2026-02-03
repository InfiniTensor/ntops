import torch

import ntops
from ntops.torch.utils import _cached_make


def glu(input, dim=-1):
    ndim = input.ndim
    if dim < 0:
        dim = ndim + dim

    dim_size = input.size(dim)
    out_shape = list(input.shape)
    out_shape[dim] //= 2
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    block_size = 1024

    kernel = _cached_make(
        ntops.kernels.glu.premake, ndim, dim, dim_size, input.dtype, block_size
    )

    kernel(input, output, dim_size)
    return output

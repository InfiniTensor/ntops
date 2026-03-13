import torch
import ntops
from ntops.torch.utils import _cached_make


def hardsigmoid(input, inplace=False):
    kernel = _cached_make(
        ntops.kernels.hardsigmoid.premake,
        input.ndim,
        inplace=inplace,
        dtype=input.dtype,
        block_size=1024,
    )

    if inplace:
        kernel(input)
        return input
    else:
        output = torch.empty_like(input)
        kernel(input, output)
        return output

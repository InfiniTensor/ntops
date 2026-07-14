import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dims(dims, ndim):
    if isinstance(dims, int):
        dims = (dims,)

    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)

    assert all(0 <= dim < ndim for dim in dims), "`dims` out of range."

    result = []
    for dim in dims:
        if dim not in result:
            result.append(dim)

    return tuple(result)


def flip(input, dims):
    dims = _normalize_dims(dims, input.ndim)

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.flip.premake,
        input.ndim,
        dims,
    )

    kernel(
        input,
        output,
    )

    return output
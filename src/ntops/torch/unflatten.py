import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim

    assert 0 <= dim < ndim

    return dim


def _normalize_sizes(sizes, dim_size):
    if isinstance(sizes, int):
        sizes = (sizes,)

    sizes = tuple(sizes)

    infer_index = None
    known_product = 1

    for i, size in enumerate(sizes):
        size = int(size)

        if size == -1:
            assert infer_index is None
            infer_index = i
        else:
            assert size >= 0
            known_product *= size

    sizes = list(sizes)

    if infer_index is not None:
        assert known_product != 0
        assert dim_size % known_product == 0
        sizes[infer_index] = dim_size // known_product
    else:
        assert math.prod(sizes) == dim_size

    return tuple(sizes)


def unflatten(input, dim, sizes):
    ndim = input.ndim

    assert ndim > 0

    dim = _normalize_dim(dim, ndim)

    sizes = _normalize_sizes(sizes, input.shape[dim])

    output_shape = (
        tuple(input.shape[:dim])
        + tuple(sizes)
        + tuple(input.shape[dim + 1:])
    )

    output = torch.empty(
        output_shape,
        dtype=input.dtype,
        device=input.device,
    )

    if output.numel() == 0:
        return output

    kernel = _cached_make(
        ntops.kernels.unflatten.premake,
        input.ndim,
        output.ndim,
    )

    kernel(input, output)

    return output
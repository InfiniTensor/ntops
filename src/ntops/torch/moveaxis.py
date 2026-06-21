import torch

import ntops
from ntops.torch.utils import _cached_make


def _to_tuple(x):
    if isinstance(x, int):
        return (x,)

    return tuple(x)


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim

    assert 0 <= dim < ndim

    return dim


def _make_permutation(ndim, source, destination):
    source = _to_tuple(source)
    destination = _to_tuple(destination)

    assert len(source) == len(destination)

    source = tuple(_normalize_dim(dim, ndim) for dim in source)
    destination = tuple(_normalize_dim(dim, ndim) for dim in destination)

    assert len(set(source)) == len(source)
    assert len(set(destination)) == len(destination)

    permutation = [dim for dim in range(ndim) if dim not in source]

    for dst, src in sorted(zip(destination, source)):
        permutation.insert(dst, src)

    return tuple(permutation)


def moveaxis(input, source, destination):
    ndim = input.ndim

    assert ndim > 0

    permutation = _make_permutation(ndim, source, destination)

    output_shape = tuple(input.shape[dim] for dim in permutation)

    output = torch.empty(
        output_shape,
        dtype=input.dtype,
        device=input.device,
    )

    if output.numel() == 0:
        return output

    kernel = _cached_make(
        ntops.kernels.moveaxis.premake,
        input.ndim,
        permutation,
    )

    kernel(input, output)

    return output
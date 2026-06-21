import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dims(dim, ndim):
    if dim is None:
        return tuple(range(ndim))

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    normalized = []
    for d in dims:
        d = int(d)
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise IndexError("dim out of range")
        if d in normalized:
            raise ValueError("dim contains duplicate values")
        normalized.append(d)

    return tuple(normalized)


def _output_shape(input_shape, reduce_dims):
    return tuple(
        size
        for axis, size in enumerate(input_shape)
        if axis not in reduce_dims
    )


def _dim_cache_key(dim):
    if dim is None:
        return None

    if isinstance(dim, int):
        return (int(dim),)

    # 关键修复：list -> tuple，避免 _cached_make 报 unhashable type: 'list'
    return tuple(int(d) for d in dim)


def count_nonzero(input, dim=None):
    reduce_dims = _normalize_dims(dim, input.ndim)
    output_shape = _output_shape(tuple(input.shape), reduce_dims)

    actual_output_shape = output_shape if len(output_shape) > 0 else (1,)

    output = torch.empty(
        actual_output_shape,
        dtype=torch.int64,
        device=input.device,
    )

    dim_key = _dim_cache_key(dim)

    kernel = _cached_make(
        ntops.kernels.count_nonzero.premake,
        tuple(input.shape),
        dim_key,
    )

    kernel(input, output)

    if len(output_shape) == 0:
        if hasattr(output, "reshape"):
            return output.reshape(())

        return output

    return output
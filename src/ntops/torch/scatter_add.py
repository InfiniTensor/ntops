import torch

import ntops
from ntops.torch.utils import _cached_make


_MAX_NTOPS_DIM_SIZE = 64


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim

    assert 0 <= dim < ndim, "`dim` out of range."

    return dim


def scatter_add(input, dim, index, src, *, out=None):
    assert input.ndim == index.ndim == src.ndim, (
        "`input`, `index`, and `src` must have the same ndim."
    )
    assert index.shape == src.shape, "`index` and `src` must have the same shape."
    assert index.dtype == torch.int64, "`index` must be torch.int64."
    assert src.dtype == input.dtype, "`src` and `input` must have the same dtype."

    dim = _normalize_dim(dim, input.ndim)

    if input.shape != index.shape:
        result = torch.scatter_add(
            input,
            dim,
            index,
            src,
        )

        if out is not None:
            out.copy_(result)
            return out

        return result

    dim_size = int(input.shape[dim])

    if dim_size > _MAX_NTOPS_DIM_SIZE:
        result = torch.scatter_add(
            input,
            dim,
            index,
            src,
        )

        if out is not None:
            out.copy_(result)
            return out

        return result

    if out is None:
        output = torch.empty_like(input)
    else:
        output = out

    kernel = _cached_make(
        ntops.kernels.scatter_add.premake,
        input.ndim,
        dim,
    )

    kernel(
        input,
        index,
        src,
        output,
    )

    return output
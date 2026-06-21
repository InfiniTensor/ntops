import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise IndexError("dim out of range")

    return dim


def _normalize_start_end(start, end, size):
    if start is None:
        start = 0

    if end is None:
        end = size

    if start < 0:
        start += size

    if end < 0:
        end += size

    start = max(0, min(start, size))
    end = max(0, min(end, size))

    return start, end


def slice_scatter(input, source, dim=0, start=None, end=None, step=1):
    if input.ndim == 0:
        raise RuntimeError("slice_scatter does not support zero-dimensional input")

    if step is None:
        step = 1

    if step <= 0:
        raise ValueError("slice_scatter only supports step > 0")

    dim = _normalize_dim(dim, input.ndim)
    start, end = _normalize_start_end(start, end, input.shape[dim])
    output = torch.empty_like(input)

    copy_kernel = _cached_make(
        ntops.kernels.slice_scatter.premake_copy,
        input.ndim,
    )

    scatter_kernel = _cached_make(
        ntops.kernels.slice_scatter.premake_scatter,
        input.ndim,
        dim,
        start,
        end,
        step,
    )

    # 第一步：output = input
    copy_kernel(input, output)

    # 第二步：output[..., start:end:step, ...] = source
    scatter_kernel(source, output)

    return output
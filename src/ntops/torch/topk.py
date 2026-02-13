import torch

import ntops
from ntops.torch.utils import _cached_make


def topk(input, k, dim=-1, largest=True, sorted=True):
    if not largest:
        raise AssertionError("Only largest=True is supported.")

    if not sorted:
        raise AssertionError("Only sorted=True is supported.")

    if dim != -1:
        raise AssertionError("Only dim=-1 is supported.")

    assert 0 < k <= input.shape[dim], "`k` must be in (0, input.shape[dim]]."

    output_shape = list(input.shape)
    output_shape[dim] = k

    values = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    indices = torch.empty(output_shape, device=input.device, dtype=torch.int64)

    dim_size = input.shape[dim]

    kernel = _cached_make(ntops.kernels.topk.premake, input.ndim, dim, k)

    kernel(input, dim_size, k, values, indices)

    return values, indices

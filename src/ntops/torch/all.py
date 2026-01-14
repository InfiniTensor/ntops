import torch

import ntops
from ntops.torch.utils import _cached_make


def next_power_of_2(n):
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()


def get_optimal_block_size(dim_size):
    target_size = next_power_of_2(dim_size)
    if target_size > 1024:
        target_size = 1024
    if target_size < 32:
        target_size = 32
    return target_size


def all(
    input, dim: int | tuple[int] | list[int] | None = None, keepdim=False, *, out=None
):
    output_dtype = torch.bool

    if dim is None:
        dims = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    if len(dims) == 0:
        if out is not None:
            out.copy_(input)
            return out
        return input.clone()

    if len(dims) > 1:
        res = input
        sorted_dims = sorted(dims, reverse=True)
        for d in sorted_dims:
            res = all(res, dim=d, keepdim=True)

        if dim is None:
            res = res.view(())
        elif not keepdim:
            for d in sorted_dims:
                res = res.squeeze(d)

        if out is not None:
            out.copy_(res)
            return out
        return res

    target_dim = dims[0] % input.ndim

    if keepdim:
        output_shape = list(input.shape)
        output_shape[target_dim] = 1
    else:
        output_shape = list(input.shape)
        output_shape.pop(target_dim)

    if out is not None:
        values = out
    else:
        values = torch.empty(output_shape, dtype=output_dtype, device=input.device)

    values_keepdim_shape = list(input.shape)
    values_keepdim_shape[target_dim] = 1
    values_for_kernel = values.view(values_keepdim_shape)

    kernel_ndim = input.ndim
    reduction_size = input.shape[target_dim]
    block_size = get_optimal_block_size(reduction_size)

    kernel = _cached_make(
        ntops.kernels.all.premake, kernel_ndim, target_dim, block_size
    )

    kernel(input, values_for_kernel)

    if dim is None:
        result = values.view(())
        if out is not None and values.data_ptr() != out.data_ptr():
            out.copy_(result)
        return result

    return values

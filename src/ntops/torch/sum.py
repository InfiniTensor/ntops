import math

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


def sum(
    input,
    dim: int | tuple[int] | list[int] | None = None,
    keepdim=False,
    *,
    dtype=None,
    out=None,
):
    if dtype is None:
        dtype = input.dtype

    if dim is None:
        current = input
        block_size = get_optimal_block_size(current.numel())

        while current.numel() > 1:
            output_len = math.ceil(current.numel() / block_size)
            output = torch.empty((output_len,), dtype=dtype, device=current.device)

            kernel = _cached_make(
                ntops.kernels.sum.premake_all_elements,
                current.ndim,
                current.dtype,
                block_size,
            )
            kernel(current, output)
            current = output

        result = current.view(())

        if out is not None:
            out.copy_(result)
            return out

        return result
    else:
        if isinstance(dim, int):
            dims = (dim,)
        else:
            dims = tuple(dim)

        output_shape = list(input.shape)
        for d in dims:
            if d < 0:
                d += input.ndim
            output_shape[d] = 1

        temp_out = torch.empty(output_shape, dtype=dtype, device=input.device)
        block_size = get_optimal_block_size(output_shape[dims[0]])

        kernel = _cached_make(
            ntops.kernels.sum.premake, input.ndim, dims, dtype, block_size
        )
        kernel(input, temp_out)

        if not keepdim:
            dims_to_remove = sorted(
                [d if d >= 0 else d + input.ndim for d in dims], reverse=True
            )

            final_shape = list(output_shape)
            for d in dims_to_remove:
                del final_shape[d]

            if not final_shape:
                temp_out = temp_out.view(())
            else:
                temp_out = temp_out.view(final_shape)

        if out is not None:
            out.copy_(temp_out)
            return out

        return temp_out

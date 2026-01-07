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


def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    dtype = input.dtype
    indices_dtype = torch.int64

    if dim is None:
        input_logic = input.contiguous().flatten()
        target_dim = 0
        original_output_shape = (k,)
    else:
        input_logic = input
        if dim < 0:
            dim += input.ndim
        target_dim = dim
        original_output_shape = list(input.shape)
        original_output_shape[dim] = k

    dim_size = input_logic.shape[target_dim]
    block_size = get_optimal_block_size(dim_size)

    if out is not None:
        values, indices = out
        if dim is None:
            values_logic = values.view(-1)
            indices_logic = indices.view(-1)
        else:
            values_logic = values
            indices_logic = indices
    else:
        logic_output_shape = list(input_logic.shape)
        logic_output_shape[target_dim] = k
        values_logic = torch.empty(logic_output_shape, dtype=dtype, device=input.device)
        indices_logic = torch.empty(
            logic_output_shape, dtype=indices_dtype, device=input.device
        )

    kernel = _cached_make(
        ntops.kernels.topk.premake,
        input_logic.ndim,
        target_dim,
        k,
        largest,
        sorted,
        dtype,
        indices_dtype,
        block_size,
    )

    kernel(input_logic, values_logic, indices_logic, k, largest)

    if out is None:
        if dim is None:
            return values_logic.view(original_output_shape), indices_logic.view(
                original_output_shape
            )
        else:
            return values_logic, indices_logic

    return values, indices

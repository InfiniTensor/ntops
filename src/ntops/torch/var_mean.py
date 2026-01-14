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


def var_mean(input, dim=None, correction=1, keepdim=False, *, dtype=None, out=None):
    if dtype is not None and input.dtype != dtype:
        input = input.to(dtype)

    ndim = input.ndim
    if dim is None:
        target_dims = tuple(range(ndim))
    elif isinstance(dim, int):
        target_dims = (dim,)
    else:
        target_dims = tuple(dim)

    target_dims = tuple(d if d >= 0 else d + ndim for d in target_dims)

    non_target_dims = [i for i in range(ndim) if i not in target_dims]
    permuted_order = non_target_dims + list(target_dims)

    input_permuted = input.permute(permuted_order).contiguous()

    num_non_target = len(non_target_dims)
    new_target_dims = tuple(range(num_non_target, ndim))

    num_elements = 1
    for d in new_target_dims:
        num_elements *= input_permuted.shape[d]

    kernel_out_shape = list(input_permuted.shape)
    for d in new_target_dims:
        kernel_out_shape[d] = 1

    temp_var = torch.empty(kernel_out_shape, dtype=input.dtype, device=input.device)
    temp_mean = torch.empty(kernel_out_shape, dtype=input.dtype, device=input.device)
    block_size = get_optimal_block_size(num_elements)

    kernel = _cached_make(
        ntops.kernels.var_mean.premake,
        input_permuted.ndim,
        new_target_dims,
        input_permuted.dtype,
        block_size,
    )

    kernel(input_permuted, temp_var, temp_mean, num_elements, correction)

    if keepdim:
        final_shape = list(input.shape)
        for d in target_dims:
            final_shape[d] = 1
    else:
        result_shape = [input.shape[i] for i in non_target_dims]
        final_shape = result_shape

    res_var = temp_var.view(final_shape) if final_shape else temp_var.view([])
    res_mean = temp_mean.view(final_shape) if final_shape else temp_mean.view([])

    if out is not None:
        out_var, out_mean = out
        out_var.copy_(res_var)
        out_mean.copy_(res_mean)
        return out_var, out_mean

    return res_var, res_mean

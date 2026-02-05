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


def binary_cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction="mean",
    out=None,
):
    if size_average is not None or reduce is not None:
        if reduce is False:
            reduction = "none"
        elif size_average is True or size_average is None:
            reduction = "mean"
        else:
            reduction = "sum"

    device = input.device
    dtype = input.dtype
    numel = input.numel()

    if target.shape != input.shape:
        target = target.expand(input.shape)

    has_weight = False
    if weight is not None:
        has_weight = True
        if weight.shape != input.shape:
            weight = weight.expand(input.shape)
        weight = weight.contiguous()
    else:
        weight = input

    compute_dtype = dtype
    if reduction != "none":
        compute_dtype = torch.float32

    if out is not None and reduction == "none":
        output_tensor = out
    else:
        output_tensor = torch.empty(input.shape, dtype=compute_dtype, device=device)

    # 高精度执行
    block_size = 1024
    kernel_bce = _cached_make(
        ntops.kernels.binary_cross_entropy.premake_bce,
        input.ndim,
        compute_dtype,
        has_weight,
        block_size,
    )
    kernel_bce(input, target, weight, output_tensor, has_weight)

    if reduction == "none":
        return output_tensor

    # Float32
    current = output_tensor.contiguous().view((numel,))

    def iterative_reduce(curr_tensor):
        while curr_tensor.numel() > 1:
            curr_numel = curr_tensor.numel()
            block_size = get_optimal_block_size(curr_numel)

            output_len = math.ceil(curr_numel / block_size)
            output = torch.empty((output_len,), dtype=compute_dtype, device=device)

            kernel_reduce = _cached_make(
                ntops.kernels.binary_cross_entropy.premake_reduce,
                compute_dtype,
                block_size,
            )
            kernel_reduce(curr_tensor, output)
            curr_tensor = output
        return curr_tensor

    final_sum_tensor = iterative_reduce(current)

    if reduction == "sum":
        # Div Kernel (div by 1.0)
        if dtype != compute_dtype:
            result = torch.empty((1,), dtype=dtype, device=device)
            kernel_cast = _cached_make(
                ntops.kernels.binary_cross_entropy.premake_div, 1.0, dtype
            )
            kernel_cast(final_sum_tensor, result, 1)
            final_sum_tensor = result

        result = final_sum_tensor.view(())
        if out is not None:
            out.copy_(result)
            return out
        return result

    elif reduction == "mean":
        final_output = torch.empty((1,), dtype=dtype, device=device)

        kernel_div = _cached_make(
            ntops.kernels.binary_cross_entropy.premake_div, numel, dtype
        )
        kernel_div(final_sum_tensor, final_output, numel)

        result = final_output.view(())
        if out is not None:
            out.copy_(result)
            return out
        return result

    return output_tensor

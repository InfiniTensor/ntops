import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement_bce(input, target, weight, output, has_weight, block_size):
    input_t = input.flatten().tile((block_size,))
    target_t = target.flatten().tile((block_size,))
    weight_t = weight.flatten().tile((block_size,))
    output_t = output.flatten().tile((block_size,))
    return input_t, target_t, weight_t, output_t, has_weight


def application_bce(input, target, weight, output, has_weight):
    val_input = ntl.cast(input, ntl.float32)
    val_target = ntl.cast(target, ntl.float32)

    eps = 1e-12
    term1 = ntl.maximum(val_input, eps)
    term2 = ntl.maximum(1.0 - val_input, eps)

    term_1 = val_target * ntl.log(term1)
    term_2 = (1.0 - val_target) * ntl.log(term2)
    loss = 0.0 - (term_1 + term_2)

    if has_weight:
        val_weight = ntl.cast(weight, ntl.float32)
        loss = loss * val_weight

    output = ntl.cast(loss, output.dtype)


def premake_bce(ndim, dtype=None, has_weight=False, block_size=None):
    arrangement_ = functools.partial(arrangement_bce, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # input
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # target
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # weight
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # output
        Tensor(0, constexpr=True, value=has_weight),  # has_weight
    )
    return arrangement_, application_bce, tensors


def arrangement_reduce(input, output, block_size):
    input_t = input.tile((block_size,))
    output_t = output.tile((1,))
    return input_t, output_t


def application_reduce(input, output):
    accumulator = 0.0
    for i in range(input.shape[0]):
        accumulator += ntl.cast(input[i], ntl.float32)
    output[0] = ntl.cast(accumulator, output.dtype)


def premake_reduce(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_reduce, block_size=block_size)
    tensors = (
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # output
    )
    return arrangement_, application_reduce, tensors


def arrangement_div(input, output, divisor):
    return input.tile((1,)), output.tile((1,)), divisor


def application_div(input, output, divisor):
    val = ntl.cast(input, ntl.float32)
    res = val / divisor
    output = ntl.cast(res, output.dtype)


def premake_div(divisor, dtype=None):
    arrangement_ = functools.partial(arrangement_div)
    tensors = (
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # output
        Tensor(0, constexpr=True, value=divisor),
    )
    return arrangement_, application_div, tensors

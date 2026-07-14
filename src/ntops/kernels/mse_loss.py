import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as element_wise_arrangement
from ntops.kernels.reduction import arrangement as reduction_arrangement


REDUCTION_NONE = 0
REDUCTION_MEAN = 1
REDUCTION_SUM = 2


def reduction_all_arrangement(input, target, output, inv_numel=None, block_size=None):
    input_arranged, target_arranged = reduction_arrangement(
        input,
        target,
        dim=tuple(range(input.ndim)),
        block_size=block_size,
    )

    output_arranged = output.tile((1,))

    if inv_numel is None:
        return input_arranged, target_arranged, output_arranged

    return input_arranged, target_arranged, output_arranged, inv_numel


def application_none(input, target, output):
    diff = input - target
    output = diff * diff  # noqa: F841


def application_sum(input, target, output):
    dtype = output.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype

    acc = ntl.cast(0, acc_dtype)

    for i in range(input.shape[0]):
        diff = ntl.cast(input[i] - target[i], acc_dtype)
        acc += ntl.sum(diff * diff)

    output = ntl.cast(acc, dtype)  # noqa: F841


def application_mean(input, target, output, inv_numel):
    dtype = output.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype

    acc = ntl.cast(0, acc_dtype)

    for i in range(input.shape[0]):
        diff = ntl.cast(input[i] - target[i], acc_dtype)
        acc += ntl.sum(diff * diff)

    output = ntl.cast(acc * inv_numel, dtype)  # noqa: F841


def premake(
    ndim,
    reduction=REDUCTION_MEAN,
    dtype=None,
    block_size=None,
):
    if reduction == REDUCTION_NONE:
        arrangement_ = functools.partial(
            element_wise_arrangement,
            block_size=block_size,
        )

        tensors = (
            Tensor(ndim, dtype=dtype),
            Tensor(ndim, dtype=dtype),
            Tensor(ndim, dtype=dtype),
        )

        return arrangement_, application_none, tensors

    assert reduction in (
        REDUCTION_MEAN,
        REDUCTION_SUM,
    ), "`reduction` must be 0, 1, or 2."

    arrangement_ = functools.partial(
        reduction_all_arrangement,
        block_size=block_size,
    )

    input = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    target = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    output = Tensor(
        1,
        dtype=dtype,
        shape_options=(
            {"constexpr": True, "upper_bound": 1},
        ),
    )
    output.shape = (1,)

    if reduction == REDUCTION_SUM:
        tensors = (
            input,
            target,
            output,
        )
        return arrangement_, application_sum, tensors

    inv_numel = Tensor(0, dtype=ninetoothed.float64)

    tensors = (
        input,
        target,
        output,
        inv_numel,
    )

    return arrangement_, application_mean, tensors
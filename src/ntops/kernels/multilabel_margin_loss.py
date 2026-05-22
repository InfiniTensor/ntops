import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement as reduction_arrangement


REDUCTION_MEAN = 1
REDUCTION_SUM = 2


def arrangement(input, target, output, block_size=None):
    input_arranged, target_arranged = reduction_arrangement(
        input,
        target,
        dim=-1,
        block_size=1,
    )

    output_arranged = output.tile((1,))

    return input_arranged, target_arranged, output_arranged


def application(input, target, output):
    dtype = output.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype

    class_size = input.shape[0]

    zero = ntl.cast(input[0] * 0, acc_dtype)
    one = zero + ntl.cast(1, acc_dtype)

    zero_t = target[0] * 0

    true = zero_t == zero_t
    false = zero_t != zero_t

    loss = zero
    alive_j = true

    for j in range(input.shape[0]):
        y_j = target[j]

        y_nonneg = y_j >= zero_t
        j_active = alive_j & y_nonneg

        x_y = zero

        # x_y = input[y_j]
        for c in range(input.shape[0]):
            c_t = zero_t + c
            x_c = ntl.cast(input[c], acc_dtype)
            x_y = ntl.where(y_j == c_t, x_c, x_y)

        for i in range(input.shape[0]):
            i_t = zero_t + i
            x_i = ntl.cast(input[i], acc_dtype)

            alive_q = true
            is_positive_i = false

            for q in range(input.shape[0]):
                t_q = target[q]

                q_nonneg = t_q >= zero_t
                q_active = alive_q & q_nonneg

                is_positive_i = is_positive_i | (q_active & (t_q == i_t))
                alive_q = alive_q & q_nonneg

            margin = one - x_y + x_i

            term = ntl.where(margin > zero, margin, zero)
            term = ntl.where(j_active, term, zero)
            term = ntl.where(is_positive_i, zero, term)

            loss += term

        alive_j = alive_j & y_nonneg

    inv_c = ntl.cast(1.0 / class_size, acc_dtype)
    loss = loss * inv_c

    output = ntl.cast(ntl.sum(loss), dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    assert ndim == 2, "`multilabel_margin_loss` kernel only supports 2D input [N, C]."

    arrangement_ = functools.partial(
        arrangement,
        block_size=1,
    )

    input = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    target = Tensor(
        ndim,
        dtype=ninetoothed.int64,
        shape_options={"constexpr": True},
    )

    output = Tensor(
        ndim - 1,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    tensors = (
        input,
        target,
        output,
    )

    return arrangement_, application, tensors


def reduce_arrangement(input, output, inv_numel=None, block_size=None):
    input_arranged = reduction_arrangement(
        input,
        dim=tuple(range(input.ndim)),
        block_size=block_size,
    )[0]

    output_arranged = output.tile((1,))

    if inv_numel is None:
        return input_arranged, output_arranged

    return input_arranged, output_arranged, inv_numel


def application_reduce_sum(input, output):
    dtype = output.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype

    acc = ntl.cast(0, acc_dtype)

    for i in range(input.shape[0]):
        acc += ntl.sum(ntl.cast(input[i], acc_dtype))

    output = ntl.cast(acc, dtype)  # noqa: F841


def application_reduce_mean(input, output, inv_numel):
    dtype = output.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype

    acc = ntl.cast(0, acc_dtype)

    for i in range(input.shape[0]):
        acc += ntl.sum(ntl.cast(input[i], acc_dtype))

    output = ntl.cast(acc * inv_numel, dtype)  # noqa: F841


def premake_reduce(
    ndim,
    reduction=REDUCTION_MEAN,
    dtype=None,
    block_size=None,
):
    assert reduction in (
        REDUCTION_MEAN,
        REDUCTION_SUM,
    ), "`reduction` must be REDUCTION_MEAN or REDUCTION_SUM."

    arrangement_ = functools.partial(
        reduce_arrangement,
        block_size=block_size,
    )

    input = Tensor(
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
            output,
        )

        return arrangement_, application_reduce_sum, tensors

    inv_numel = Tensor(0, dtype=ninetoothed.float64)

    tensors = (
        input,
        output,
        inv_numel,
    )

    return arrangement_, application_reduce_mean, tensors
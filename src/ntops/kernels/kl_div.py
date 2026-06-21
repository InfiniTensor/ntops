import functools

from ninetoothed import Tensor
import ninetoothed.language as ntl


REDUCTION_SUM = 0
REDUCTION_BATCHMEAN = 1
REDUCTION_MEAN = 2


def arrangement(
    input,
    target,
    output,
    batch_size,
    feature_size,
    reduction,
    log_target,
    block_size=None,
):
    return (
        input,
        target,
        output,
        batch_size,
        feature_size,
        reduction,
        log_target,
    )


def application(
    input,
    target,
    output,
    batch_size,
    feature_size,
    reduction,
    log_target,
):
    acc = ntl.zeros((), dtype=ntl.float32)

    for i in range(batch_size):
        for j in range(feature_size):
            x = input[i, j].to(ntl.float32)
            t = target[i, j].to(ntl.float32)

            if log_target:
                # PyTorch:
                # loss = exp(target) * (target - input)
                loss = ntl.exp(t) * (t - x)
            else:
                # PyTorch kl_div(log_target=False) uses xlogy(target, target)
                # semantics:
                #
                #   target == 0 -> target * log(target) is treated as 0
                #
                # Directly computing t * log(t) gives NaN when t == 0,
                # because 0 * -inf = NaN.
                is_zero = t == 0
                safe_t = ntl.where(is_zero, 1.0, t)

                loss = ntl.where(
                    is_zero,
                    0.0,
                    t * (ntl.log(safe_t) - x),
                )

            acc += loss

    # 0: sum
    # 1: batchmean
    # 2: mean
    if reduction == 1:
        acc = acc / batch_size
    elif reduction == 2:
        acc = acc / (batch_size * feature_size)

    output[0] = acc  # noqa: F841


def premake(
    input_shape,
    reduction=REDUCTION_BATCHMEAN,
    log_target=False,
    dtype=None,
    block_size=None,
):
    assert len(input_shape) == 2, "kl_div currently only supports 2-D input"

    batch_size_value = int(input_shape[0])
    feature_size_value = int(input_shape[1])
    reduction_value = int(reduction)
    log_target_value = bool(log_target)

    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    input = Tensor(2, dtype=dtype)
    target = Tensor(2, dtype=dtype)
    output = Tensor(1, dtype=dtype)

    input.shape = tuple(input_shape)
    target.shape = tuple(input_shape)
    output.shape = (1,)

    batch_size = Tensor(0, constexpr=True, value=batch_size_value)
    feature_size = Tensor(0, constexpr=True, value=feature_size_value)
    reduction = Tensor(0, constexpr=True, value=reduction_value)
    log_target = Tensor(0, constexpr=True, value=log_target_value)

    tensors = (
        input,
        target,
        output,
        batch_size,
        feature_size,
        reduction,
        log_target,
    )

    return arrangement_, application, tensors
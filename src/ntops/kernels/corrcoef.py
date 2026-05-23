import functools

from ninetoothed import Tensor
import ninetoothed.language as ntl


def arrangement_1d(input, output, input_size, block_size=None):
    return input, output, input_size


def arrangement_2d_single_row(input, output, num_cols, block_size=None):
    return input, output, num_cols


def arrangement_2d(input, output, num_rows, num_cols, block_size=None):
    return input, output, num_rows, num_cols


def application_1d(input, output, input_size):
    # torch.corrcoef(1D tensor) returns scalar 1.
    # Use output shape (1,) to avoid 0-dim output pointer issue.
    output[0] = 1.0  # noqa: F841


def application_2d_single_row(input, output, num_cols):
    # torch.corrcoef(input) for input shape (1, N) returns scalar 1.
    # Kernel still writes shape (1,), wrapper squeezes it to scalar.
    output[0] = 1.0  # noqa: F841


def application_2d(input, output, num_rows, num_cols):
    correction = num_cols - 1

    for i in range(num_rows):
        for j in range(num_rows):
            if num_cols <= 1:
                output[i, j] = float("nan")  # noqa: F841

            elif i == j:
                output[i, j] = 1.0  # noqa: F841

            elif j < i:
                output[i, j] = output[j, i]  # noqa: F841

            else:
                mean_i = ntl.zeros((), dtype=ntl.float32)
                mean_j = ntl.zeros((), dtype=ntl.float32)

                for k in range(num_cols):
                    mean_i += input[i, k].to(ntl.float32)
                    mean_j += input[j, k].to(ntl.float32)

                mean_i = mean_i / num_cols
                mean_j = mean_j / num_cols

                cov = ntl.zeros((), dtype=ntl.float32)
                var_i = ntl.zeros((), dtype=ntl.float32)
                var_j = ntl.zeros((), dtype=ntl.float32)

                for k in range(num_cols):
                    xi = input[i, k].to(ntl.float32) - mean_i
                    xj = input[j, k].to(ntl.float32) - mean_j

                    cov += xi * xj
                    var_i += xi * xi
                    var_j += xj * xj

                cov = cov / correction
                var_i = var_i / correction
                var_j = var_j / correction

                denom = var_i * var_j

                inv_sqrt = ntl.rsqrt(denom)

                # Improve rsqrt precision.
                inv_sqrt = inv_sqrt * (1.5 - 0.5 * denom * inv_sqrt * inv_sqrt)
                inv_sqrt = inv_sqrt * (1.5 - 0.5 * denom * inv_sqrt * inv_sqrt)

                corr = cov * inv_sqrt
                corr = ntl.minimum(ntl.maximum(corr, -1.0), 1.0)

                output[i, j] = corr  # noqa: F841


def premake(
    input_shape,
    dtype=None,
    block_size=None,
):
    input_shape = tuple(input_shape)

    if len(input_shape) == 1:
        input_size_value = int(input_shape[0])

        arrangement = functools.partial(
            arrangement_1d,
            block_size=block_size,
        )

        input = Tensor(1, dtype=dtype)
        output = Tensor(1, dtype=dtype)

        input.shape = input_shape
        output.shape = (1,)

        input_size = Tensor(0, constexpr=True, value=input_size_value)

        tensors = (
            input,
            output,
            input_size,
        )

        return arrangement, application_1d, tensors

    if len(input_shape) == 2:
        num_rows_value = int(input_shape[0])
        num_cols_value = int(input_shape[1])

        input = Tensor(2, dtype=dtype)
        input.shape = input_shape

        if num_rows_value == 1:
            arrangement = functools.partial(
                arrangement_2d_single_row,
                block_size=block_size,
            )

            output = Tensor(1, dtype=dtype)
            output.shape = (1,)

            num_cols = Tensor(0, constexpr=True, value=num_cols_value)

            tensors = (
                input,
                output,
                num_cols,
            )

            return arrangement, application_2d_single_row, tensors

        arrangement = functools.partial(
            arrangement_2d,
            block_size=block_size,
        )

        output = Tensor(2, dtype=dtype)

        output.shape = (num_rows_value, num_rows_value)

        num_rows = Tensor(0, constexpr=True, value=num_rows_value)
        num_cols = Tensor(0, constexpr=True, value=num_cols_value)

        tensors = (
            input,
            output,
            num_rows,
            num_cols,
        )

        return arrangement, application_2d, tensors
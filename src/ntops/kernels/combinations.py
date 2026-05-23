import functools
import math

import ninetoothed
from ninetoothed import Tensor


def _num_combinations(n, r, with_replacement):
    if r < 0:
        raise ValueError("r must be non-negative")

    if r == 0:
        return 1

    if n == 0:
        return 0

    if with_replacement:
        return math.comb(n + r - 1, r)

    if r > n:
        return 0

    return math.comb(n, r)


def arrangement(input, output, input_size, r, with_replacement, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    return input, output, input_size, r, with_replacement


def application(input, output, input_size, r, with_replacement):
    if r == 0:
        return

    if r == 1:
        row = 0

        for i in range(input_size):
            output[row, 0] = input[i]  # noqa: F841
            row += 1

    elif r == 2:
        row = 0

        if with_replacement:
            for i in range(input_size):
                for j in range(i, input_size):
                    output[row, 0] = input[i]  # noqa: F841
                    output[row, 1] = input[j]  # noqa: F841
                    row += 1
        else:
            for i in range(input_size):
                for j in range(i + 1, input_size):
                    output[row, 0] = input[i]  # noqa: F841
                    output[row, 1] = input[j]  # noqa: F841
                    row += 1

    elif r == 3:
        row = 0

        if with_replacement:
            for i in range(input_size):
                for j in range(i, input_size):
                    for k in range(j, input_size):
                        output[row, 0] = input[i]  # noqa: F841
                        output[row, 1] = input[j]  # noqa: F841
                        output[row, 2] = input[k]  # noqa: F841
                        row += 1
        else:
            for i in range(input_size):
                for j in range(i + 1, input_size):
                    for k in range(j + 1, input_size):
                        output[row, 0] = input[i]  # noqa: F841
                        output[row, 1] = input[j]  # noqa: F841
                        output[row, 2] = input[k]  # noqa: F841
                        row += 1


def premake(
    input_size,
    r=2,
    with_replacement=False,
    dtype=None,
    block_size=None,
):
    input_size = int(input_size)
    r = int(r)
    with_replacement = bool(with_replacement)

    if r < 0:
        raise ValueError("r must be non-negative")

    if r > 3:
        raise NotImplementedError("combinations currently only supports r <= 3")

    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    input = Tensor(1, dtype=dtype)
    output = Tensor(2, dtype=dtype)

    input_size_tensor = Tensor(0, constexpr=True, value=input_size)
    r_tensor = Tensor(0, constexpr=True, value=r)
    with_replacement_tensor = Tensor(0, constexpr=True, value=with_replacement)

    num_rows = _num_combinations(
        n=input_size,
        r=r,
        with_replacement=with_replacement,
    )

    input.shape = (input_size,)
    output.shape = (num_rows, r)

    tensors = (
        input,
        output,
        input_size_tensor,
        r_tensor,
        with_replacement_tensor,
    )

    return arrangement_, application, tensors
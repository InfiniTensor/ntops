import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, nan_val, posinf_val, neginf_val, output):
    # Detect special values using IEEE 754 properties
    is_nan = input != input  # NaN is the only value not equal to itself
    is_posinf = input == float("+inf")
    is_neginf = input == float("-inf")

    # Replace using arithmetic: result = input * !special + replacement * special
    # Use ntl.where with same-shaped tensors for type compatibility
    result = ntl.where(is_nan, nan_val, input)
    result = ntl.where(is_posinf, posinf_val, result)
    output = ntl.where(is_neginf, neginf_val, result)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),  # input
        Tensor(ndim, dtype=dtype),  # nan_val (broadcast to input shape)
        Tensor(ndim, dtype=dtype),  # posinf_val
        Tensor(ndim, dtype=dtype),  # neginf_val
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, application, tensors

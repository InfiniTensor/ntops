import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, output_var, output_mean, num_elements, correction):
    acc_sum = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        acc_sum += ntl.cast(input[i], ntl.float32)

    n_float = ntl.cast(num_elements, ntl.float32)
    mean = ntl.sum(acc_sum, 0) / n_float

    acc_sq_diff = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        val = ntl.cast(input[i], ntl.float32)
        diff = val - mean
        mask = input[i].offsets(-1) < num_elements
        diff = ntl.where(mask, diff, 0)
        acc_sq_diff += diff * diff

    sum_sq_diff = ntl.sum(acc_sq_diff, 0)

    divisor = ntl.cast(num_elements - correction, ntl.float32)
    var = sum_sq_diff / divisor

    output_var[0] = ntl.cast(var, output_var.dtype.dtype)
    output_mean[0] = ntl.cast(mean, output_mean.dtype.dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(0, dtype=dtype),
    )
    return arrangement_, application, tensors

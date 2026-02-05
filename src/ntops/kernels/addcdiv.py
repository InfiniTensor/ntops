import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, tensor1, tensor2, value, output):
    dtype = output.dtype
    val_input = ntl.cast(input, dtype)
    val_t1 = ntl.cast(tensor1, dtype)
    val_t2 = ntl.cast(tensor2, dtype)
    val_v = ntl.cast(value, dtype)

    # out = input + value * (t1 / t2)
    res = val_input + val_v * (val_t1 / val_t2)

    output = res


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

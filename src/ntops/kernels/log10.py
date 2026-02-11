import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    if input.dtype == ntl.float16:
        output = ntl.log(ntl.cast(input, ntl.float32)) * 0.4342944819032518
    else:
        output = ntl.log(input) * 0.4342944819032518


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

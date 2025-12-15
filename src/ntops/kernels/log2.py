import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    dtype = input.dtype
    log2_dtype = dtype if dtype != ntl.float16 else ntl.float32
    output = ntl.cast(ntl.log2(ntl.cast(input, log2_dtype)), dtype)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

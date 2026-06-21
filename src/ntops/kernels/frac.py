import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    # frac(x) = x - trunc(x); trunc rounds toward zero. floor only accepts
    # fp32/fp64, so compute in fp32 and cast back (ceil(x) = -floor(-x) avoids
    # the int-range issues of an int cast).
    # frac(x) = x - trunc(x); trunc rounds toward zero. floor only accepts
    # fp32/fp64, so compute in fp32 and cast back (ceil(x) = -floor(-x) avoids
    # the int-range issues of an int cast). element_wise is single-level so
    # `output.dtype` is the element type.
    x = ntl.cast(input, ntl.float32)
    trunc = ntl.where(x >= 0, ntl.floor(x), -ntl.floor(-x))
    output = ntl.cast(x - trunc, output.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

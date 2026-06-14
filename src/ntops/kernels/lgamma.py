import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(input, output):
    # libdevice.lgamma computes the natural logarithm of the absolute value of the gamma function
    # Cast to float32 for computation (lgamma supports float32/float64)
    # The result will be automatically cast back to the correct dtype
    input_f32 = ntl.cast(input, ntl.float32)

    output = libdevice.lgamma(input_f32)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

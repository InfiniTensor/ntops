import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(x, y, output):
    # libdevice.nextafter returns the next representable floating-point value
    # Cast inputs to float32 for computation (nextafter supports float32/float64)
    # The result will be automatically cast back to the correct dtype
    x_f32 = ntl.cast(x, ntl.float32)
    y_f32 = ntl.cast(y, ntl.float32)

    output = libdevice.nextafter(x_f32, y_f32)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

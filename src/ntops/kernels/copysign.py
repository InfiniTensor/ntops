import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(x, y, output):
    # libdevice.copysign only supports float32 and float64
    # Cast inputs to float32 for computation
    x_f32 = ntl.cast(x, ntl.float32)
    y_f32 = ntl.cast(y, ntl.float32)

    # The result will be automatically cast back to the correct dtype
    output = libdevice.copysign(x_f32, y_f32)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

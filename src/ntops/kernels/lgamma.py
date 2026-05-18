import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 8192


def application(input, output):
    output = libdevice.lgamma(input)  # noqa: F841


def half_application(input, output):
    output = ntl.cast(libdevice.lgamma(ntl.cast(input, ntl.float32)), ntl.float16)  # noqa: F841


def premake(ndim, half=False, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    application_ = half_application if half else application

    return arrangement_, application_, tensors

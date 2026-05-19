import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application_float32_compute(input, output):
    # libdevice.lgamma only supports fp32/fp64; cast narrower floats up.
    dtype = output.dtype
    output = ntl.cast(  # noqa: F841
        libdevice.lgamma(ntl.cast(input, ntl.float32)),
        dtype,
    )


def application_native(input, output):
    dtype = output.dtype
    output = ntl.cast(  # noqa: F841
        libdevice.lgamma(input),
        dtype,
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if dtype in (ninetoothed.float16, ninetoothed.bfloat16):
        application = application_float32_compute
    else:
        application = application_native

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

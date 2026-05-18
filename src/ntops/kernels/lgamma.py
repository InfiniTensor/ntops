import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(input, output):
    # libdevice.lgamma only supports fp32/fp64; cast narrower floats up.
    dtype = output.dtype
    compute_dtype = (
        dtype
        if dtype != ntl.float16 and dtype != ntl.bfloat16
        else ntl.float32
    )
    output = ntl.cast(  # noqa: F841
        libdevice.lgamma(ntl.cast(input, compute_dtype)),
        dtype,
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

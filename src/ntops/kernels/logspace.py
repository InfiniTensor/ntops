import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(output, start, step_val, base_val):
    pid = ntl.program_id(0)
    j = ntl.arange(0, output.shape[0])
    idx = pid * output.shape[0] + j
    # Compute exponent in float32 for precision
    exponent = (
        ntl.cast(start, ntl.float32)
        + ntl.cast(idx, ntl.float32) * ntl.cast(step_val, ntl.float32)
    )
    # Compute base^exponent in float32, then cast to output dtype
    result = libdevice.pow(ntl.cast(base_val, ntl.float32), exponent)
    output = ntl.cast(result, output.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
    )

    return arrangement_, application, tensors

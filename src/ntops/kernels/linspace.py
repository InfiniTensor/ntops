import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(output, start, step_val):
    pid = ntl.program_id(0)
    j = ntl.arange(0, output.shape[0])
    idx = pid * output.shape[0] + j
    # Compute in float32 for intermediate precision, then cast to output dtype
    result = (
        ntl.cast(start, ntl.float32)
        + ntl.cast(idx, ntl.float32) * ntl.cast(step_val, ntl.float32)
    )
    output = ntl.cast(result, output.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
    )

    return arrangement_, application, tensors

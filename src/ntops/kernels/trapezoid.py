import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(areas, output):
    dtype = output.dtype.dtype
    total = ntl.cast(0, ntl.float32)

    for i in range(areas.shape[0]):
        total = total + ntl.cast(ntl.sum(areas[i]), ntl.float32)

    for j in range(output.shape[0]):
        output[j] = ntl.cast(total, dtype)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=-1, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

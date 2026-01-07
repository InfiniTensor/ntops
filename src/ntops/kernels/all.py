import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, output):
    val_block = input[0]
    bool_block = val_block != 0
    res = ntl.min(bool_block, axis=0)
    output[0] = res


def premake(ndim, dim, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    tensors = (
        Tensor(ndim, other=1),
        Tensor(ndim, dtype="int8"),
    )
    return arrangement_, application, tensors

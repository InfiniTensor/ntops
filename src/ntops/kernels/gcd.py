import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    a = ntl.abs(ntl.cast(input, ntl.int64))
    b = ntl.abs(ntl.cast(other, ntl.int64))

    while ntl.max(ntl.cast(b != 0, ntl.int32)) == 1:
        mask = b != 0
        safe_b = ntl.where(mask, b, 1)
        r = a % safe_b
        a = ntl.where(mask, b, a)
        b = ntl.where(mask, r, b)

        mask = b != 0
        safe_b = ntl.where(mask, b, 1)
        r = a % safe_b
        a = ntl.where(mask, b, a)
        b = ntl.where(mask, r, b)

        mask = b != 0
        safe_b = ntl.where(mask, b, 1)
        r = a % safe_b
        a = ntl.where(mask, b, a)
        b = ntl.where(mask, r, b)

        mask = b != 0
        safe_b = ntl.where(mask, b, 1)
        r = a % safe_b
        a = ntl.where(mask, b, a)
        b = ntl.where(mask, r, b)

    output = ntl.cast(a, output.dtype)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

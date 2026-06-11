import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(a, b, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    a_arranged = a.tile((block_size, 1))
    a_arranged = a_arranged.expand((-1, 2))

    b_arranged = b.tile((block_size, 1))
    b_arranged = b_arranged.expand((-1, 2))

    output_arranged = output.tile((block_size, 2))

    return a_arranged, b_arranged, output_arranged


def application(a, b, output):
    col_idx = ntl.arange(0, output.shape[1])
    output = ntl.where(col_idx == 0, a, b)  # noqa: F841


def premake(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors

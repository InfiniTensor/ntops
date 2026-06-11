import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size as _block_size


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = _block_size()

    input_arranged = input.tile((1, block_size))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    output_arranged = output.tile((1,))

    return input_arranged, output_arranged


def application(input, output):
    n = input.shape[0]
    best_val = ntl.cast(0, ntl.float32)
    best_count = ntl.cast(-1, ntl.int32)
    for i in range(n):
        val = input[i]
        ok = val == val
        count = ntl.cast(0, ntl.int32)
        for j in range(n):
            vj = input[j]
            same = ntl.where(vj == val, ntl.cast(1, ntl.int32), ntl.cast(0, ntl.int32))
            same = ntl.where(vj != vj, ntl.cast(0, ntl.int32), same)
            count = count + same
        better = (count >= best_count) & ok
        best_val = ntl.where(better, val, best_val)
        best_count = ntl.where(better, count, best_count)
    output = best_val


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("nan")),
        Tensor(1, dtype=dtype),
    )

    return arrangement_, application, tensors

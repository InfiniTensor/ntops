import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(condition, input, other, output):
    # condition 非 0 当 True，语义对齐 torch.where
    cond_bool = condition != 0
    output = ntl.where(cond_bool, input, other)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),  # condition
        Tensor(ndim, dtype=dtype),  # input
        Tensor(ndim, dtype=dtype),  # other
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, application, tensors
import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    fp32_input = (input * 2.0) / 2.0
    floor_val = ntl.math.floor(fp32_input)
    ceil_val = ntl.math.ceil(fp32_input)
    
    # 获取截断后的整数部分
    trunc_val = ntl.where(fp32_input >= 0, floor_val, ceil_val)
    output = input - trunc_val  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    input_f32 = ntl.cast(input, ntl.float32)
    sp = ntl.log(1 + ntl.exp(input_f32))
    exp_sp = ntl.exp(sp)
    exp_neg_sp = ntl.exp(-sp)
    tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
    result = ntl.cast(input_f32 * tanh_sp, input.dtype)
    output = result  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors

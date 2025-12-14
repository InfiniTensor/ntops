import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement

def arrangement_dot_full(input, tensor, out, block_size):
    # input/tensor: (N, )
    # output: (1, )
    input = input.tile((-1, )) # (1, ), dtype=(block_size, )
    tensor = tensor.tile((-1, )) # (1, ), dtype=(block_size, )
    out = out.tile((1, )) # (1, ), dtype=(1, )
    return input, tensor, out

def application_dot_full(input, tensor, out):
    out = ntl.sum(input * tensor)

def premake_dot_full(dtype, block_size):
    arrangement_ = functools.partial(arrangement_dot_full, block_size=block_size)

    tensors = (
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),
        Tensor(1, dtype=dtype)
    )

    return arrangement_, application_dot_full, tensors


# ========= 分块计算 =========

def arrangement_dot_divide(input, tensor, out_temp, block_size):
    # input/tensor: (N, )
    # output: (N // block_size, )
    input = input.tile((block_size, )) # (N // block_size, block_size), dtype=(block_size, )
    tensor = tensor.tile((block_size, )) # (N // block_size, block_size), dtype=(block_size, )
    out_temp = out_temp.tile((1, )) # (N // block_size, ), dtype=(1, )
    return input, tensor, out_temp

def application_dot_divide(input, tensor, out_temp):
    out_temp = ntl.sum(input * tensor, 0)

def arrangement_dot_conquer(input_block_wise, out, block_size):
    # input/tensor: (N // block_size, )
    # output: (1, )
    input_block_wise = input_block_wise.tile((-1, )) # (1, ), dtype=(block_size, )
    out = out.tile((1, )) # (1, ), dtype=(1, )
    return input_block_wise, out

def application_dot_conquer(input_block_wise, out):
    out = ntl.sum(input_block_wise)

def premake_dot_divide(dtype, block_size):
    arrangement_ = functools.partial(arrangement_dot_divide, block_size=block_size)

    tensors = (
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),
        Tensor(1, dtype=dtype)
    )

    return arrangement_, application_dot_divide, tensors

def premake_dot_conquer(dtype, block_size):
    arrangement_ = functools.partial(arrangement_dot_conquer, block_size=block_size)

    tensors = (
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),
        Tensor(1, dtype=dtype)
    )

    return arrangement_, application_dot_conquer, tensors

import torch

import ntops
import math
from ntops.torch.utils import _cached_make


def dot(input, other, *, out=None):
    assert input.ndim == 1 and other.ndim == 1

    if out is None:
        out = torch.empty((1, ), dtype=input.dtype, device=input.device)

    input_numel = input.numel()
    if input_numel <= 4096:
        block_size = 1 << (input_numel - 1).bit_length()
        kernel = _cached_make(ntops.kernels.dot.premake_dot_full, dtype=input.dtype, block_size=block_size)
        kernel(input, other, out)
        out = out.view(())
    else:
        sqrt_n = math.isqrt(input_numel)
        block_size = 1 << (sqrt_n - 1).bit_length()
        temp_out = torch.empty(((input_numel // block_size), ), dtype=input.dtype, device=input.device)

        kernel1 = _cached_make(ntops.kernels.dot.premake_dot_divide, dtype=input.dtype, block_size=block_size)
        kernel1(input, other, temp_out)

        kernel2 = _cached_make(ntops.kernels.dot.premake_dot_conquer, dtype=input.dtype, block_size=block_size)
        kernel2(temp_out, out)

        out = out.view(())

    return out

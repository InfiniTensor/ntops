import torch

import ntops
from ntops.torch.utils import _cached_make


def index_select(input, dim, index, *, out=None):
    assert index.ndim == 1, "Index tensor must be 1-dimensional."

    T = input.shape[dim]
    T_pow2 = 1 << (T - 1).bit_length()
    S = index.shape[0]
    S_pow2 = 1 << (S - 1).bit_length()

    if dim < 0:
        dim += input.ndim

    if out is None:
        output_shape = list(input.shape)
        output_shape[dim] = index.shape[0]
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 256
    kernel = _cached_make(ntops.kernels.index_select.premake, input.ndim, dim, T_pow2=T_pow2, S_pow2=S_pow2, block_size=block_size)
    kernel(input, out, index, T, S)

    return out

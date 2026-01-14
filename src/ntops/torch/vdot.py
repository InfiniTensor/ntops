import torch

import ntops
from ntops.torch.utils import _cached_make


def vdot(input, other, *, out=None):
    assert input.ndim == 1 and other.ndim == 1
    assert input.shape[0] == other.shape[0]

    if out is None:
        out = torch.empty((), dtype=input.dtype, device=input.device)

    # 创建一个临时的 1D tensor 作为 accumulator
    accumulator = torch.empty((1,), dtype=input.dtype, device=input.device)
    
    kernel = _cached_make(ntops.kernels.vdot.premake, input.ndim)
    kernel(input, other, accumulator)
    
    # 从 accumulator 提取标量值
    out.copy_(accumulator[0])
    return out
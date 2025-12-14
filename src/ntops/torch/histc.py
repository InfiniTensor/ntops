
import torch

import ntops
from ntops.torch.utils import _cached_make
import builtins

def histc(input, bins=100, min=None, max=None, is_moore=False):
    if min is None:
        min = torch.min(input).item()

    if max is None:
        max = torch.max(input).item()

    # block_size = builtins.min(1024, 1 << (input.shape[0] - 1).bit_length())
    # block_size = builtins.max(32, block_size)
    block_size = 256
    # 初始化输出为零，因为我们会累加直方图计数
    num_bins_pow2 = 1 << (bins - 1).bit_length()  # 计算大于等于 bins 的最小 2 的幂次方
    out = torch.empty((bins,), dtype=input.dtype, device=input.device)
    out = torch.nn.init.zeros_(out)

    if is_moore:
        kernel = _cached_make(ntops.kernels.histc.premake_manual, input.dtype, block_size=block_size)
    else:
        kernel = _cached_make(ntops.kernels.histc.premake_builtin, input.dtype, block_size=block_size)

    kernel(input, out, min, max, num_bins_pow2)

    return out


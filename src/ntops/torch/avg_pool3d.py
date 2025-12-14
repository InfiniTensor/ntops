import math
import torch

import ntops
from ntops.torch.utils import _cached_make

def _effective_counts(input_shape, kernel_size, stride):
    """Compute number of valid elements per output position when padding is implicit.

    For ceil_mode=True the last window in each dimension may be smaller than the
    kernel. We precompute the effective element count so we can rescale the
    zero-padded average produced by the kernel back to `count_include_pad=False`
    semantics to match PyTorch.
    """

    N, C, D_in, H_in, W_in = input_shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride

    D_out = math.ceil((D_in - kd) / sd + 1)
    H_out = math.ceil((H_in - kh) / sh + 1)
    W_out = math.ceil((W_in - kw) / sw + 1)

    d_range = torch.arange(D_out, device="cuda")
    h_range = torch.arange(H_out, device="cuda")
    w_range = torch.arange(W_out, device="cuda")

    kd_eff = torch.clamp(D_in - d_range * sd, max=kd).clamp_min(0)
    kh_eff = torch.clamp(H_in - h_range * sh, max=kh).clamp_min(0)
    kw_eff = torch.clamp(W_in - w_range * sw, max=kw).clamp_min(0)

    counts = kd_eff[:, None, None] * kh_eff[None, :, None] * kw_eff[None, None, :]
    counts = counts.view(1, 1, D_out, H_out, W_out).expand(N, C, -1, -1, -1)

    return counts


def avg_pool3d(input, kernel_size: int | tuple[int, int, int], stride: None | int | tuple[int, int, int] = None, ceil_mode=False):
    assert input.ndim == 5 or input.ndim == 4, "Input tensor must be 4-dimensional (N, C, D_in, H_in, W_in) or 3-dimensional (C, D_in, H_in, W_in)"

    if input.ndim == 4:
        input = input.unsqueeze(0)  # 添加 batch 维度

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if stride is None:
        stride = kernel_size

    # 计算输出长度
    N, C, D_in, H_in, W_in = input.shape
    if ceil_mode:
        D_out = math.ceil((D_in - kernel_size[0]) / stride[0] + 1)
        H_out = math.ceil((H_in - kernel_size[1]) / stride[1] + 1)
        W_out = math.ceil((W_in - kernel_size[2]) / stride[2] + 1)
    else:
        D_out = math.floor((D_in - kernel_size[0]) / stride[0] + 1)
        H_out = math.floor((H_in - kernel_size[1]) / stride[1] + 1)
        W_out = math.floor((W_in - kernel_size[2]) / stride[2] + 1)

    output_shape = (N, C, D_out, H_out, W_out)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.avg_pool3d.premake,
        input.ndim,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        stride[0],
        stride[1],
        stride[2],
        block_size=block_size,
        ceil_mode=ceil_mode,
        dtype=input.dtype
    )
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    kernel(input, output, kernel_volume)

    if ceil_mode:
        counts = _effective_counts((N, C, D_in, H_in, W_in), kernel_size, stride)
        counts = counts.to(dtype=output.dtype, device=output.device)
        torch.mul(output, kernel_volume, out=output)
        torch.div(output, counts, out=output)
        # output.mul_(kernel_volume).div_(counts)

    return output

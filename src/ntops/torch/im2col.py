import torch

import ntops
from ntops.torch.pooling import _calculate_output_size
from ntops.torch.utils import _cached_make


def im2col(
    input,
    kernel_size,
    dilation=1,
    padding=0,
    stride=1,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    n, c, h, w = input.shape

    h_ = _calculate_output_size(
        h,
        kernel_size[0],
        stride=stride[0],
        padding=padding[0],
        dilation=dilation[0],
    )

    w_ = _calculate_output_size(
        w,
        kernel_size[1],
        stride=stride[1],
        padding=padding[1],
        dilation=dilation[1],
    )

    output_2d = torch.empty(
        (
            n * h_ * w_,
            c * kernel_size[0] * kernel_size[1],
        ),
        dtype=input.dtype,
        device=input.device,
    )

    kernel = _cached_make(
        ntops.kernels.im2col.premake,
        kernel_size_h=kernel_size[0],
        kernel_size_w=kernel_size[1],
        stride_h=stride[0],
        stride_w=stride[1],
        padding_h=padding[0],
        padding_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        ceil_mode=False,
        block_size_m=32,
        block_size_n=32,
    )

    kernel(input, output_2d)

    output = output_2d.reshape(
        n,
        h_ * w_,
        c * kernel_size[0] * kernel_size[1],
    )

    output = output.permute(0, 2, 1)

    return output
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed.language import libdevice
from ninetoothed import Tensor
from ninetoothed import Symbol

def arrangement(
    *tensors,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    block_size,
    ceil_mode,
):
    input, output, kernel_volume = tensors
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, D_in, H_in, W_in) output: (N, C, D_out, H_out, W_out)
    input_arranged = input.tile(
        (1, 1, kernel_size_d, kernel_size_h, kernel_size_w),
        (1, 1, stride_d, stride_h, stride_w),
        floor_mode=not ceil_mode,
    )
    # => (N, C, D_out, H_out, W_out), dtype=(1, 1, k_d, k_h, k_w)
    input_arranged = input_arranged.ravel()
    # => (N, C, D_out, H_out, W_out, 1, 1, k_d, k_h, k_w)
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    # => (N*C*D_out*H_out*W_out, k_d*k_h*k_w)

    # k_d*k_h*k_w 的找到最近的 2 的倍数
    nearest_pow2 = 1 << (kernel_size_d * kernel_size_h * kernel_size_w - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., k_d*k_h*k_w // nearest_pow2 = 1), dtype=(1, nearest_pow2)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    # => (..., 1), dtype=(nearest_pow2, )
    input_arranged = input_arranged.tile((block_size, -1))
    # => (..., 1), dtype=(block_size, 1), dtype=(nearest_pow2, )
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    output_arranged = output.tile((1, 1, 1, 1, 1))
    # => (N, C, D_out, H_out, W_out), dtype=(1, 1, 1, 1, 1)
    output_arranged = output_arranged.ravel()
    # => (N, C, D_out, H_out, W_out, 1, 1, 1, 1)
    output_arranged = output_arranged.flatten(end_dim=5).flatten(start_dim=1)
    # => (N*C*D_out*H_out*W_out, 1)
    output_arranged = output_arranged.tile((block_size, -1))
    # => (..., 1), dtype=(block_size, 1)
    output_arranged.dtype = output_arranged.dtype.squeeze(1)
    # => (..., 1), dtype=(block_size, )

    return input_arranged, output_arranged, kernel_volume


def application(input, output, kernel_volume):
    # input:    (block_size, nearest_pow2)
    # output:   (block_size,)

    # Input 数据: (block_size, nearest_pow2)
    # 这是实际的像素值，越界处填充为 0
    val_sum = ntl.sum(input, axis=1)  # (block_size, )
    output = val_sum / kernel_volume  # (block_size, )


def premake(
    ndim,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    block_size=None,
    ceil_mode=False,
    dtype=None,
):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_d=kernel_size_d,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),  # input
        Tensor(ndim, dtype=dtype),  # output
        Tensor(0, dtype=dtype),  # kernel_volume
    )

    return arrangement_, application, tensors

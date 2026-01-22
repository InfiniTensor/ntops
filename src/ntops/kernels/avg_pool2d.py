import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = ninetoothed.block_size()

KERNEL_SIZE_H = Symbol("kernel_size_h", constexpr=True, upper_bound=16)
KERNEL_SIZE_W = Symbol("kernel_size_w", constexpr=True, upper_bound=16)
STRIDE_H = Symbol("stride_h", constexpr=True)
STRIDE_W = Symbol("stride_w", constexpr=True)
PADDING_H = Symbol("padding_h", constexpr=True)
PADDING_W = Symbol("padding_w", constexpr=True)
DILATION_H = Symbol("dilation_h", constexpr=True)
DILATION_W = Symbol("dilation_w", constexpr=True)


def arrangement(
    input,
    output,
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    ceil_mode=None,
    block_size=None,
):
    if kernel_size_h is None:
        kernel_size_h = KERNEL_SIZE_H

    if kernel_size_w is None:
        kernel_size_w = KERNEL_SIZE_W

    if stride_h is None:
        stride_h = STRIDE_H

    if stride_w is None:
        stride_w = STRIDE_W

    if padding_h is None:
        padding_h = PADDING_H

    if padding_w is None:
        padding_w = PADDING_W

    if dilation_h is None:
        dilation_h = DILATION_H

    if dilation_w is None:
        dilation_w = DILATION_W

    if ceil_mode is None:
        ceil_mode = False

    if block_size is None:
        block_size = BLOCK_SIZE

    input_arranged = input.pad(
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )
    input_arranged = input_arranged.tile(
        (1, 1, kernel_size_h, kernel_size_w),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=not ceil_mode,
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.sum(input, axis=-1) / input.shape[-1]  # noqa: F841


def premake(
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    ceil_mode=None,
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        ceil_mode=ceil_mode,
        block_size=block_size,
    )

    tensors = (Tensor(4, dtype=dtype), Tensor(4, dtype=dtype))

    return arrangement_, application, tensors

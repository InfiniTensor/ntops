import functools

from ninetoothed import Symbol, Tensor


BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32


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
    block_size_m=None,
    block_size_n=None,
):
    if kernel_size_h is None:
        kernel_size_h = Symbol("kernel_size_h", constexpr=True, upper_bound=8)

    if kernel_size_w is None:
        kernel_size_w = Symbol("kernel_size_w", constexpr=True, upper_bound=8)

    if stride_h is None:
        stride_h = Symbol("stride_h", constexpr=True)

    if stride_w is None:
        stride_w = Symbol("stride_w", constexpr=True)

    if padding_h is None:
        padding_h = Symbol("padding_h", constexpr=True)

    if padding_w is None:
        padding_w = Symbol("padding_w", constexpr=True)

    if dilation_h is None:
        dilation_h = Symbol("dilation_h", constexpr=True)

    if dilation_w is None:
        dilation_w = Symbol("dilation_w", constexpr=True)

    if ceil_mode is None:
        ceil_mode = False

    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N
    input_arranged = input.pad(
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )

    input_arranged = input_arranged.tile(
        (1, input.shape[1], kernel_size_h, kernel_size_w),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=not ceil_mode,
    )

    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    input_arranged = input_arranged.tile((block_size_m, block_size_n))

    # output: (N * OH * OW, C * KH * KW)
    output_arranged = output.tile((block_size_m, block_size_n))

    return input_arranged, output_arranged


def application(input, output):
    output = input


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
    block_size_m=None,
    block_size_n=None,
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
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    input = Tensor(
        4,
        dtype=dtype,
        shape_options=(
            {"upper_bound": 16},     # N
            {"upper_bound": 64},     # C
            {"upper_bound": 256},    # H
            {"upper_bound": 256},    # W
        ),
    )

    output = Tensor(
        2,
        dtype=dtype,
        shape_options=(
            {"upper_bound": 65536},  # N * OH * OW
            {"upper_bound": 1024},   # C * KH * KW
        ),
    )

    tensors = (
        input,
        output,
    )

    return arrangement_, application, tensors
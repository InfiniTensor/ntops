import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


SCALE_BLOCK_SIZE = 128
NUM_WARPS = (4, 8)
NUM_STAGES = (1, 2)

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=16, upper_bound=128)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=16, upper_bound=128)
MIN_SPECIALIZED_BLOCK_SIZE = 16
MAX_SPECIALIZED_BLOCK_SIZE_M = 64


class ScalingVariant(enum.IntEnum):
    BLOCK_WISE_1X128_128X128 = enum.auto()

    BLOCK_WISE_1X128_1X128 = enum.auto()

    BLOCK_WISE_128X128_1X128 = enum.auto()


def _largest_divisible_block_size(extent, maximum=SCALE_BLOCK_SIZE):
    block_size = maximum

    while block_size > MIN_SPECIALIZED_BLOCK_SIZE and extent % block_size:
        block_size //= 2

    return block_size


def specialized_block_sizes(scaling_variant, m, n):
    """Choose concrete power-of-two tiles for a shape-specialized kernel."""
    block_size_m = (
        MIN_SPECIALIZED_BLOCK_SIZE
        if m == 1
        else SCALE_BLOCK_SIZE
        if scaling_variant == ScalingVariant.BLOCK_WISE_128X128_1X128
        else _largest_divisible_block_size(m, MAX_SPECIALIZED_BLOCK_SIZE_M)
    )
    block_size_n = (
        SCALE_BLOCK_SIZE
        if scaling_variant == ScalingVariant.BLOCK_WISE_1X128_128X128
        else _largest_divisible_block_size(n)
    )

    return block_size_m, block_size_n


def _block_sizes(scaling_variant, block_size_m, block_size_n):
    if block_size_m is None:
        block_size_m = (
            SCALE_BLOCK_SIZE
            if scaling_variant == ScalingVariant.BLOCK_WISE_128X128_1X128
            else BLOCK_SIZE_M
        )

    if block_size_n is None:
        block_size_n = (
            SCALE_BLOCK_SIZE
            if scaling_variant == ScalingVariant.BLOCK_WISE_1X128_128X128
            else BLOCK_SIZE_N
        )

    return block_size_m, block_size_n


def arrangement(
    input,
    mat2,
    scale_a,
    scale_b,
    output,
    scaling_variant,
    block_size_m=None,
    block_size_n=None,
    input_block_size_m=None,
):
    block_size_m, block_size_n = _block_sizes(
        scaling_variant,
        block_size_m,
        block_size_n,
    )

    if input_block_size_m is None:
        input_block_size_m = block_size_m

    output_arranged = output.tile((block_size_m, block_size_n))

    singleton_broadcast = input_block_size_m == 1 and block_size_m > 1
    input_tile_m = block_size_m if singleton_broadcast else input_block_size_m
    input_arranged = input.tile(
        (input_tile_m, SCALE_BLOCK_SIZE),
        dilation=(0, 1) if singleton_broadcast else None,
    )
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    mat2_arranged = mat2.tile((SCALE_BLOCK_SIZE, block_size_n))
    mat2_arranged = mat2_arranged.tile((-1, 1))
    mat2_arranged = mat2_arranged.expand((output_arranged.shape[0], -1))
    mat2_arranged.dtype = mat2_arranged.dtype.squeeze(1)

    if scaling_variant == ScalingVariant.BLOCK_WISE_128X128_1X128:
        scale_a_arranged = scale_a.tile((1, 1))
    else:
        scale_a_tile_m = block_size_m if singleton_broadcast else input_block_size_m
        scale_a_arranged = scale_a.tile(
            (scale_a_tile_m, 1),
            dilation=(0, 1) if singleton_broadcast else None,
        )

    scale_a_arranged = scale_a_arranged.tile((1, -1))
    scale_a_arranged = scale_a_arranged.expand((-1, output_arranged.shape[1]))
    scale_a_arranged.dtype = scale_a_arranged.dtype.squeeze(0)

    if scaling_variant == ScalingVariant.BLOCK_WISE_1X128_128X128:
        scale_b_arranged = scale_b.tile((1, 1))
    else:
        scale_b_arranged = scale_b.tile((1, block_size_n))

    scale_b_arranged = scale_b_arranged.tile((-1, 1))
    scale_b_arranged = scale_b_arranged.expand((output_arranged.shape[0], -1))
    scale_b_arranged.dtype = scale_b_arranged.dtype.squeeze(1)

    return (
        input_arranged,
        mat2_arranged,
        scale_a_arranged,
        scale_b_arranged,
        output_arranged,
    )


def bias_arrangement(
    input,
    mat2,
    scale_a,
    scale_b,
    bias,
    output,
    scaling_variant,
    block_size_m=None,
    block_size_n=None,
    input_block_size_m=None,
):
    block_size_m, block_size_n = _block_sizes(
        scaling_variant,
        block_size_m,
        block_size_n,
    )
    arranged = arrangement(
        input,
        mat2,
        scale_a,
        scale_b,
        output,
        scaling_variant,
        block_size_m,
        block_size_n,
        input_block_size_m,
    )
    if input_block_size_m is None:
        input_block_size_m = block_size_m

    bias_arranged = bias.tile((input_block_size_m, block_size_n))

    return (*arranged[:-1], bias_arranged, arranged[-1])


def _accumulate(input, mat2, scale_a, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        # Operand legalization is a target capability handled by the backend.
        partial = ntl.dot(input[k], mat2[k])
        block_scale = scale_a[k] * scale_b[k]
        accumulator += partial * block_scale

    return accumulator


def _accumulate_singleton(input, mat2, scale_a, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        partial = ntl.dot(input[k], mat2[k])
        block_scale = scale_a[k] * scale_b[k]
        accumulator += partial * block_scale

    return accumulator


def application(input, mat2, scale_a, scale_b, output):
    output = _accumulate(input, mat2, scale_a, scale_b, output)


def bias_application(input, mat2, scale_a, scale_b, bias, output):
    output = _accumulate(input, mat2, scale_a, scale_b, output) + bias


def singleton_application(input, mat2, scale_a, scale_b, output):
    output = _accumulate_singleton(input, mat2, scale_a, scale_b, output)


def singleton_bias_application(input, mat2, scale_a, scale_b, bias, output):
    output = _accumulate_singleton(input, mat2, scale_a, scale_b, output) + bias


def premake(
    scaling_variant,
    input_dtype=None,
    mat2_dtype=None,
    scale_dtype=None,
    output_dtype=None,
    bias_dtype=None,
    input_shape=None,
    mat2_shape=None,
    scale_a_shape=None,
    scale_b_shape=None,
    output_shape=None,
    bias_shape=None,
    block_size_m=None,
    block_size_n=None,
):
    broadcast_singleton_m = (
        input_shape is not None
        and input_shape[0] == 1
        and block_size_m is not None
        and block_size_m > 1
    )
    # Keep singleton inputs on the ordinary masked tile path.  The backend can
    # eliminate the mask for full tiles, while singleton rows must not be
    # materialized as stride-zero broadcasts.
    input_block_size_m = block_size_m
    arrangement_function = arrangement if bias_dtype is None else bias_arrangement
    application_function = (
        singleton_application
        if broadcast_singleton_m and bias_dtype is None
        else singleton_bias_application
        if broadcast_singleton_m
        else application
        if bias_dtype is None
        else bias_application
    )
    arrangement_ = functools.partial(
        arrangement_function,
        scaling_variant=scaling_variant,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        input_block_size_m=input_block_size_m,
    )

    def tensor(dtype, shape):
        if shape is None:
            return Tensor(2, dtype=dtype)

        return Tensor(shape=tuple(shape), dtype=dtype)

    tensors = [
        tensor(input_dtype, input_shape),
        tensor(mat2_dtype, mat2_shape),
        tensor(scale_dtype, scale_a_shape),
        tensor(scale_dtype, scale_b_shape),
    ]

    if bias_dtype is not None:
        tensors.append(tensor(bias_dtype, bias_shape))

    tensors.append(tensor(output_dtype, output_shape))

    return arrangement_, application_function, tuple(tensors)

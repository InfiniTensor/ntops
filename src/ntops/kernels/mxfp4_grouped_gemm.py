"""MXFP4 W4A16 grouped expert matrix multiplication."""

import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

MXFP4_BLOCK_SIZE = 32
PACKED_VALUES_PER_BYTE = 2
PACKED_VALUES_PER_SCALE = MXFP4_BLOCK_SIZE // PACKED_VALUES_PER_BYTE

NUM_WARPS = (4, 8)
NUM_STAGES = (1,)

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=16, upper_bound=64)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=16, upper_bound=128)


class ComputeDtypeVariant(enum.IntEnum):
    FLOAT16 = enum.auto()

    BFLOAT16 = enum.auto()


def _decode_e2m1(packed):
    """Decode E2M1 nibbles to FP32 values."""
    packed_i32 = packed.to(ntl.int32)
    exponent = (packed_i32 >> 1) & 0x3
    mantissa = packed_i32 & 0x1
    sign = packed_i32 >> 3
    shift = ntl.where(exponent == 0, 0, exponent - 1)
    magnitude_i32 = ntl.where(exponent == 0, mantissa, mantissa + 2) << shift
    magnitude = magnitude_i32.to(ntl.float32) * 0.5

    return ntl.where(sign != 0, -magnitude, magnitude)


def _decode_e8m0(scale):
    """Decode an E8M0 scale byte to an FP32 multiplier."""
    scale_i32 = scale.to(ntl.int32)
    bits = ntl.where(scale_i32 == 0, 0x00400000, scale_i32 << 23)

    return ntl.cast(bits, ntl.float32, bitcast=True)


def _arrange_activation(input, output, block_size_m):
    arranged = input.tile((1, block_size_m, MXFP4_BLOCK_SIZE))
    arranged = arranged.tile((1, 1, -1))
    arranged = arranged.expand((-1, -1, output.shape[2]))
    arranged.dtype = arranged.dtype.squeeze((0, 1))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)

    return arranged


def _arrange_weight(weight_mxfp4, output, block_size_n):
    arranged = weight_mxfp4.permute((0, 2, 1))
    arranged = arranged.tile((1, PACKED_VALUES_PER_SCALE, block_size_n))
    arranged = arranged.tile((1, -1, 1))
    arranged = arranged.expand((-1, output.shape[1], -1))
    arranged.dtype = arranged.dtype.squeeze((0, 2))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)

    return arranged


def _arrange_scale(weight_scale, output, block_size_n):
    arranged = weight_scale.permute((0, 2, 1))
    arranged = arranged.tile((1, 1, block_size_n))
    arranged = arranged.tile((1, -1, 1))
    arranged = arranged.expand((-1, output.shape[1], -1))
    arranged.dtype = arranged.dtype.squeeze((0, 2))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

    return arranged


def arrangement(
    input,
    weight_mxfp4,
    weight_scale,
    output,
    block_size_m=None,
    block_size_n=None,
):
    """Map one launch grid to expert, token, and output-column tiles."""
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    return (
        _arrange_activation(input, output_arranged, block_size_m),
        _arrange_weight(weight_mxfp4, output_arranged, block_size_n),
        _arrange_scale(weight_scale, output_arranged, block_size_n),
        output_arranged,
    )


def bfloat16_application(
    input,
    weight_mxfp4,
    weight_scale,
    output,
):
    """Accumulate the grouped GEMM with BF16 dot operands."""
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k_idx in range(input.shape[0]):
        packed = weight_mxfp4[k_idx].to(ntl.int32)
        scale = _decode_e8m0(weight_scale[k_idx])
        weight_even = _decode_e2m1(packed & 0x0F).to(ntl.bfloat16)
        weight_odd = _decode_e2m1(packed >> 4).to(ntl.bfloat16)
        weight = ntl.trans(
            ntl.interleave(ntl.trans(weight_even), ntl.trans(weight_odd))
        )
        accumulator += ntl.dot(input[k_idx].to(ntl.bfloat16), weight) * scale

    output[:] = accumulator.to(output.dtype)


def float16_application(
    input,
    weight_mxfp4,
    weight_scale,
    output,
):
    """Accumulate the grouped GEMM with FP16 dot operands."""
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k_idx in range(input.shape[0]):
        packed = weight_mxfp4[k_idx].to(ntl.int32)
        scale = _decode_e8m0(weight_scale[k_idx])
        weight_even = _decode_e2m1(packed & 0x0F).to(ntl.float16)
        weight_odd = _decode_e2m1(packed >> 4).to(ntl.float16)
        weight = ntl.trans(
            ntl.interleave(ntl.trans(weight_even), ntl.trans(weight_odd))
        )
        accumulator += ntl.dot(input[k_idx].to(ntl.float16), weight) * scale

    output[:] = accumulator.to(output.dtype)


def premake(
    num_experts,
    k,
    n,
    input_dtype,
    output_dtype,
    compute_dtype,
    block_size_m=None,
    block_size_n=None,
):
    """Create the grouped jagged kernel definition."""
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    output_tensor = Tensor(
        shape=(num_experts, None, n),
        dtype=output_dtype,
        jagged_dim=1,
    )
    tensors = (
        Tensor(
            shape=(num_experts, None, k),
            dtype=input_dtype,
            jagged_dim=1,
        ),
        Tensor(shape=(num_experts, n, k // 2), dtype="uint8"),
        Tensor(
            shape=(num_experts, n, k // MXFP4_BLOCK_SIZE),
            dtype="uint8",
        ),
        output_tensor,
    )

    application_ = (
        bfloat16_application
        if compute_dtype == ComputeDtypeVariant.BFLOAT16
        else float16_application
    )

    return arrangement_, application_, tensors


__all__ = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "ComputeDtypeVariant",
    "MXFP4_BLOCK_SIZE",
    "NUM_STAGES",
    "NUM_WARPS",
    "PACKED_VALUES_PER_BYTE",
    "PACKED_VALUES_PER_SCALE",
    "arrangement",
    "bfloat16_application",
    "float16_application",
    "premake",
]

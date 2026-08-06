"""MXFP4 W4A16 Grouped Expert Matrix Multiplication Kernel.

This kernel implements high-performance grouped matrix multiplication with
MXFP4 (Micro-scaled FP4) weight quantization for Mixture-of-Experts inference.

MXFP4 format:
- 4-bit floating-point weights with E2M1 format
- Shared 8-bit scaling factor per block (16 elements per block)
- W4A16: 4-bit weights, 16-bit activations (BF16/FP16)

The kernel performs dequantization on-the-fly during matrix multiplication to
minimize memory bandwidth while maintaining numerical accuracy.
"""

import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()

MXFP4_BLOCK_SIZE = 16  # Elements per MXFP4 scale
MXFP4_MAX_VALUE = 6.0  # Max representable value in E2M1


class Mxfp4DequantMode(enum.IntEnum):
    """MXFP4 dequantization mode selection."""

    STANDARD = enum.auto()
    FAST_APPROX = enum.auto()


def _dequantize_mxfp4_block(scaled_weight, scale, output):
    """Dequantize a block of MXFP4 weights.

    Converts 4-bit E2M1 weights back to higher precision using the
    block-scale factor. The E2M1 format has:
    - 1 sign bit, 2 exponent bits, 1 mantissa bit
    - Range: [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
    """
    # Extract sign, exponent, and mantissa from 4-bit values
    sign = ntl.where(scaled_weight >> 3, -1.0, 1.0)
    exp_bits = (scaled_weight >> 1) & 0x3
    mantissa_bit = scaled_weight & 0x1

    # Compute dequantized value: sign * scale * (2^(exp-1) + mantissa*0.5)
    exp_val = ntl.bitwise_xor(exp_bits, 1)
    mantissa = ntl.cast(mantissa_bit, ntl.float32) * 0.5
    dequant_normalized = ntl.exp2(exp_val) * (1.0 + mantissa)

    # Apply per-element sign and scale
    output[:] = sign * dequant_normalized * scale


def arrangement(
    input,
    weight_mxfp4,
    weight_scale,
    expert_offsets,
    num_experts,
    output,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    """Arrange tensors for MXFP4 grouped GEMM.

    The grouped GEMM processes multiple expert computations in a single kernel
    launch by batching along the expert dimension.
    """
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M
    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N
    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    # Arrange output: (num_experts, M, N) -> tiled
    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged = output_arranged.permute((1, 0, 2, 3))
    output_arranged = output_arranged.flatten(start_dim=0, end_dim=1)
    output_arranged = output_arranged.flatten(start_dim=-2)
    output_arranged.dtype = output_arranged.dtype.squeeze((0, 2))
    output_arranged.dtype.dtype = output_arranged.dtype.dtype.squeeze((0, 1))

    # Arrange input activations: shared across experts (M, K)
    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    # Arrange MXFP4 weights: (num_experts, N, K // 2 packed)
    weight_arranged = weight_mxfp4.tile((1, block_size_n, block_size_k // 2))
    weight_arranged = weight_arranged.permute((1, 0, 2, 3))
    weight_arranged = weight_arranged.flatten(start_dim=0, end_dim=1)
    weight_arranged = weight_arranged.flatten(start_dim=-2)
    weight_arranged.dtype = weight_arranged.dtype.squeeze((0, 2))
    weight_arranged.dtype.dtype = weight_arranged.dtype.dtype.squeeze((0, 1))

    # Arrange weight scales: (num_experts, N, K // MXFP4_BLOCK_SIZE)
    scale_arranged = weight_scale.tile((1, block_size_n, block_size_k // MXFP4_BLOCK_SIZE))
    scale_arranged = scale_arranged.permute((1, 0, 2, 3))
    scale_arranged = scale_arranged.flatten(start_dim=0, end_dim=1)
    scale_arranged = scale_arranged.flatten(start_dim=-2)
    scale_arranged.dtype = scale_arranged.dtype.squeeze((0, 2))
    scale_arranged.dtype.dtype = scale_arranged.dtype.dtype.squeeze((0, 1))

    # Expert offsets for routing
    expert_offsets_arranged = expert_offsets
    num_experts_arranged = num_experts

    return (
        input_arranged,
        weight_arranged,
        scale_arranged,
        expert_offsets_arranged,
        num_experts_arranged,
        output_arranged,
    )


def application(
    input,
    weight_mxfp4,
    weight_scale,
    expert_offsets,
    num_experts,
    output,
):
    """Apply MXFP4 grouped GEMM computation.

    Performs on-the-fly dequantization of MXFP4 weights during the matrix
    multiplication to compute: output[expert] = input @ weight[expert].T
    """
    num_k_blocks = input.shape[0]

    # Accumulator for the output tile
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k_idx in range(num_k_blocks):
        # Load input tile
        input_tile = input[k_idx]

        # Process each expert's contribution in the group
        for expert_idx in range(num_experts):
            # Load packed MXFP4 weights for this expert
            weight_packed = weight_mxfp4[expert_idx]

            # Unpack and dequantize 4-bit weights on-the-fly
            # Low nibble: elements [0, 2, 4, ...]
            low_nib = weight_packed & 0x0F
            # High nibble: elements [1, 3, 5, ...]
            high_nib = (weight_packed >> 4) & 0x0F

            # Load corresponding scale for this k-block
            scale = weight_scale[expert_idx, k_idx]

            # Dequantize both nibbles
            weight_low = _dequant_mxfp4_element(low_nib, scale)
            weight_high = _dequant_mxfp4_element(high_nib, scale)

            # Interleave dequantized weights back to original order
            weight_full = _interleave_mxfp4(weight_low, weight_high)

            # Accumulate: output += input @ dequant_weight.T
            accumulator[expert_idx] += ntl.dot(input_tile, ntl.trans(weight_full))

    # Store final result
    output[:] = accumulator


def _dequant_mxfp4_element(nibble_val, scale):
    """Dequantize a single E2M1 nibble value.

    E2M1 format lookup:
    0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0,
    4: 4.0, 5: 6.0 (unused), 6: -0.0, 7: -0.5,
    8: -1.0, 9: -2.0, 10: -4.0
    """
    # Use constexpr lookup table for fast dequantization
    lut = ntl.constexpr([0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0])
    normalized = lut[nibble_val]
    return normalized * scale


def _interleave_mxfp4(low, high):
    """Interleave low and high nibble dequantized values."""
    # Pack as [low[0], high[0], low[1], high[1], ...]
    interleaved = ntl.empty((low.shape[0] * 2,), dtype=low.dtype)
    interleaved[0::2] = low
    interleaved[1::2] = high
    return interleaved


def premake(
    num_experts=None,
    m_dim=None,
    n_dim=None,
    k_dim=None,
    input_dtype=None,
    output_dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    """Premake configuration for MXFP4 grouped GEMM kernel."""
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    # Activation tensor: (M, K)
    input_tensor = Tensor(2, dtype=input_dtype)

    # Packed MXFP4 weights: (num_experts, N, K // 2)
    # K // 2 because two 4-bit weights are packed per byte
    weight_tensor = Tensor(3, dtype=ninetoothed.uint8)

    # Weight scales: (num_experts, N, K // MXFP4_BLOCK_SIZE)
    scale_tensor = Tensor(3, dtype=ninetoothed.float8_e4m3)

    # Expert routing offsets
    expert_offsets_tensor = Tensor(1, dtype=ninetoothed.int32)

    # Number of experts (constexpr)
    num_experts_tensor = Tensor(0, constexpr=True, value=num_experts)

    # Output tensor: (num_experts, M, N)
    output_tensor = Tensor(3, dtype=output_dtype)

    tensors = (
        input_tensor,
        weight_tensor,
        scale_tensor,
        expert_offsets_tensor,
        num_experts_tensor,
        output_tensor,
    )

    return arrangement_, application, tensors

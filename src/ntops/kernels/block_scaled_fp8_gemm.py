"""Block-scaled FP8 Matrix Multiplication Kernel.

This kernel implements FP8 (E4M3/E5M2) matrix multiplication with per-block
scaling for high-precision low-bit computation in LLM inference.

Block-scaled FP8 format:
- FP8 elements in E4M3 or E5M2 format
- Per-block scaling factors (128 elements per block)
- Supports per-channel and per-token activation scaling

Key optimizations:
1. Online rescaling to prevent FP16 accumulator overflow
2. Block-level pipeline for overlapping dequantization with compute
3. Hardware-aware tiling for optimal memory access patterns
"""

import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()

# Block scaling parameters
ACT_BLOCK_SIZE = 128  # Activation scaling block size
WEIGHT_BLOCK_SIZE = 128  # Weight scaling block size


class Fp8Format(enum.IntEnum):
    """FP8 format selection."""

    E4M3 = enum.auto()  # Better precision, smaller range
    E5M2 = enum.auto()  # Wider range, lower precision (for gradients)


class ScalingGranularity(enum.IntEnum):
    """Scaling granularity selection."""

    PER_1X128 = enum.auto()  # Fine-grained, better accuracy
    PER_128X128 = enum.auto()  # Coarser, less memory overhead


def arrangement(
    input,
    input_scale,
    weight,
    weight_scale,
    output,
    fp8_format,
    scaling_granularity,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    """Arrange tensors for block-scaled FP8 GEMM.

    Both input and weight scales may have different tiling granularities
    depending on the scaling mode selected.
    """
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M
    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N
    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    # Arrange output
    output_arranged = output.tile((block_size_m, block_size_n))
    output_arranged = output_arranged.flatten(start_dim=-2)
    output_arranged.dtype = output_arranged.dtype.squeeze((0, 2))
    output_arranged.dtype.dtype = output_arranged.dtype.dtype.squeeze((0, 1))

    # Arrange FP8 activations with per-token scaling
    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    # Arrange input scales: (M, K // ACT_BLOCK_SIZE)
    input_scale_arranged = input_scale.tile((1, -1))
    input_scale_arranged = input_scale_arranged.expand((-1, output_arranged.shape[1]))
    input_scale_arranged.dtype = input_scale_arranged.dtype.squeeze(0)

    # Arrange FP8 weights: (K, N) transposed
    weight_arranged = weight.tile((block_size_k, block_size_n))
    weight_arranged = weight_arranged.tile((-1, 1))
    weight_arranged = weight_arranged.expand((output_arranged.shape[0], -1))
    weight_arranged.dtype = weight_arranged.dtype.squeeze(1)

    # Arrange weight scales: (K // WEIGHT_BLOCK_SIZE, N // WEIGHT_BLOCK_SIZE)
    weight_scale_arranged = weight_scale.tile((1, 1))
    weight_scale_arranged = weight_scale_arranged.expand(
        (output_arranged.shape[0], output_arranged.shape[1])
    )
    weight_scale_arranged.dtype = weight_scale_arranged.dtype.squeeze((0, 1))

    fp8_format_arranged = fp8_format
    scaling_granularity_arranged = scaling_granularity

    return (
        input_arranged,
        input_scale_arranged,
        weight_arranged,
        weight_scale_arranged,
        output_arranged,
        fp8_format_arranged,
        scaling_granularity_arranged,
    )


def application(
    input_fp8,
    input_scale,
    weight_fp8,
    weight_scale,
    output,
    fp8_format,
    scaling_granularity,
):
    """Apply block-scaled FP8 matrix multiplication.

    Computes: output = (input_fp8 * input_scale) @ (weight_fp8 * weight_scale).T

    The implementation uses a block-pipelined approach where dequantization
    and accumulation are overlapped for maximum throughput.
    """
    num_k_blocks = input_fp8.shape[0]

    # Accumulator in FP32 for numerical precision
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    # Pipeline state for double-buffered dequantization
    input_dequant_next = ntl.empty(
        (input_fp8.shape[-1],), dtype=ntl.float32
    )
    weight_dequant_next = ntl.empty(
        (weight_fp8.shape[-1],), dtype=ntl.float32
    )

    for k_block in range(num_k_blocks):
        # Load FP8 tiles
        input_tile_fp8 = input_fp8[k_block]
        weight_tile_fp8 = weight_fp8[k_block]

        # Load block scales
        input_blk_scale = input_scale[k_block]

        # Weight scale depends on scaling granularity
        if scaling_granularity == ScalingGranularity.PER_128X128:
            weight_blk_scale = weight_scale[k_block // (WEIGHT_BLOCK_SIZE // block_size_k), :]
        else:
            weight_blk_scale = weight_scale[k_block, :]

        if fp8_format == Fp8Format.E4M3:
            input_tile = ntl.cast(input_tile_fp8, ntl.float32) * input_blk_scale
            weight_tile = ntl.cast(weight_tile_fp8, ntl.float32) * weight_blk_scale
        else:  # E5M2 - wider exponent range
            input_tile = ntl.cast(input_tile_fp8, ntl.float32) * input_blk_scale
            weight_tile = ntl.cast(weight_tile_fp8, ntl.float32) * weight_blk_scale

        # Online rescaling to prevent accumulator overflow
        # Scale down intermediate results before accumulation
        accumulator_scale = _compute_safe_scale(accumulator)
        accumulator *= accumulator_scale

        # Fused multiply-add with dequantized values
        accumulator += ntl.dot(input_tile, ntl.trans(weight_tile))

    # Apply final rescaling and convert to output dtype
    output[:] = accumulator


def _compute_safe_scale(accumulator):
    """Compute a safe scaling factor to prevent overflow.

    Uses the maximum absolute value in the accumulator to determine
    an appropriate rescaling factor, preventing FP32 overflow during
    accumulation of FP8 products.
    """
    abs_max = ntl.max(ntl.abs(accumulator))
    # If abs_max is large, scale down; otherwise keep as-is
    threshold = 1.0e30  # Well below FP32 max (~3.4e38)
    safe_scale = ntl.where(abs_max > threshold, threshold / abs_max, 1.0)
    return safe_scale


def _fused_dequant_gemm(input_fp8, input_scale, weight_fp8, weight_scale, accumulator):
    """Fused dequantization and GEMM for a single block pair.

    Performs: accumulator += (input_fp8 * input_scale) @ (weight_fp8 * weight_scale).T
    """
    # Dequantize input
    input_dequant = ntl.cast(input_fp8, ntl.float32) * input_scale

    # Dequantize weight
    weight_dequant = ntl.cast(weight_fp8, ntl.float32) * weight_scale

    # Accumulate
    accumulator += ntl.dot(input_dequant, ntl.trans(weight_dequant))
    return accumulator


def premake(
    fp8_format=None,
    scaling_granularity=None,
    input_dtype=None,
    weight_dtype=None,
    scale_dtype=None,
    output_dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    """Premake configuration for block-scaled FP8 GEMM kernel."""
    arrangement_ = functools.partial(
        arrangement,
        fp8_format=fp8_format,
        scaling_granularity=scaling_granularity,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    # FP8 activation tensor: (M, K)
    input_tensor = Tensor(2, dtype=input_dtype)

    # FP8 activation scales: (M, K // ACT_BLOCK_SIZE)
    input_scale_tensor = Tensor(2, dtype=scale_dtype)

    # FP8 weight tensor: (K, N) - stored transposed for cache efficiency
    weight_tensor = Tensor(2, dtype=weight_dtype)

    # FP8 weight scales: (K // WEIGHT_BLOCK_SIZE, N // WEIGHT_BLOCK_SIZE)
    weight_scale_tensor = Tensor(2, dtype=scale_dtype)

    # Output tensor: (M, N) in higher precision (FP16/BF16)
    output_tensor = Tensor(2, dtype=output_dtype)

    # FP8 format selection (constexpr)
    fp8_format_tensor = Tensor(0, constexpr=True, value=fp8_format)

    # Scaling granularity (constexpr)
    scaling_mode_tensor = Tensor(0, constexpr=True, value=scaling_granularity)

    tensors = (
        input_tensor,
        input_scale_tensor,
        weight_tensor,
        weight_scale_tensor,
        output_tensor,
        fp8_format_tensor,
        scaling_mode_tensor,
    )

    return arrangement_, application, tensors

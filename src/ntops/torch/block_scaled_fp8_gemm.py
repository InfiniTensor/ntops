"""Block-scaled FP8 GEMM torch wrapper."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def block_scaled_fp8_gemm(
    input_fp8,
    input_scale,
    weight_fp8,
    weight_scale,
    fp8_format=1,
    scaling_granularity=1,
):
    """Block-scaled FP8 matrix multiplication.

    Args:
        input_fp8: FP8 activations (M, K) in float8_e4m3fn.
        input_scale: Per-token activation scales (M, K//128).
        weight_fp8: FP8 weights (N, K) in float8_e4m3fn.
        weight_fp8: Per-block weight scales (K//128, N//128).
        fp8_format: FP8 format selection (1=E4M3, 2=E5M2).
        scaling_granularity: Scaling granularity (1=1x128, 2=128x128).

    Returns:
        output: Result tensor (M, N) in BF16.
    """
    m = input_fp8.shape[0]
    n = weight_fp8.shape[0]

    output = torch.empty((m, n), dtype=torch.bfloat16, device=input_fp8.device)

    kernel = _cached_make(
        ntops.kernels.block_scaled_fp8_gemm.premake,
        fp8_format=fp8_format,
        scaling_granularity=scaling_granularity,
        input_dtype=input_fp8.dtype,
        weight_dtype=weight_fp8.dtype,
        scale_dtype=input_scale.dtype,
        output_dtype=output.dtype,
    )

    kernel(
        input_fp8,
        input_scale,
        weight_fp8,
        weight_scale,
        output,
        fp8_format,
        scaling_granularity,
    )

    return output

"""MXFP4 Grouped GEMM torch wrapper."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def mxfp4_grouped_gemm(
    input,
    weight_mxfp4,
    weight_scale,
    expert_offsets,
    num_experts,
):
    """MXFP4 W4A16 grouped expert matrix multiplication.

    Args:
        input: Input activations (M, K) in BF16/FP16.
        weight_mxfp4: Packed MXFP4 weights (num_experts, N, K//2) in uint8.
        weight_scale: Per-block scaling factors (num_experts, N, K//16) in FP8.
        expert_offsets: Routing offsets for each expert (num_experts + 1,).
        num_experts: Number of experts.

    Returns:
        output: Result tensor (num_experts, M, N) in BF16.
    """
    output = torch.empty(
        (num_experts, input.shape[0], weight_mxfp4.shape[1]),
        dtype=input.dtype,
        device=input.device,
    )

    kernel = _cached_make(
        ntops.kernels.mxfp4_grouped_gemm.premake,
        num_experts=num_experts,
        input_dtype=input.dtype,
        output_dtype=output.dtype,
    )

    kernel(input, weight_mxfp4, weight_scale, expert_offsets, num_experts, output)

    return output

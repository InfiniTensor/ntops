import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def multilabel_margin_loss(input, target, reduction="mean"):
    """Multi-label margin loss — full NineToothed kernel implementation.

    Equivalent to torch.nn.functional.multilabel_margin_loss.

    Args:
        input: (C,) or (N, C) tensor of predicted values.
        target: (C,) or (N, C) tensor of target class indices, padded with -1.
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    _validate_args(input, target, reduction)

    # Ensure input is at least 2D.
    single_sample = input.dim() == 1
    if single_sample:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    N, C = input.shape
    BLOCK_C = 2 ** math.ceil(math.log2(C))

    # Kernel 1: per-batch losses, shape (N,).
    output = torch.empty(N, dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.multilabel_margin_loss.premake,
        C=C,
        BLOCK_C=BLOCK_C,
        max_num_configs=1,
    )

    kernel(input, target, output, C)

    if reduction == "none":
        result = output
        if single_sample:
            result = result.squeeze(0)
    elif N == 0:
        # Empty batch: sum=0, mean=NaN (matching PyTorch).
        result = torch.zeros(1, dtype=input.dtype, device=input.device)
        if reduction == "mean":
            result[:] = float("nan")
    else:
        # Kernel 2: atomic reduction into a scalar.
        scale = 1.0 / N if reduction == "mean" else 1.0
        result = torch.zeros(1, dtype=input.dtype, device=input.device)

        reduce_kernel = _cached_make(
            ntops.kernels.reduce_sum.premake,
            block_size=128,
            max_num_configs=1,
        )
        reduce_kernel(output, result, scale)

        if single_sample:
            result = result.squeeze()

    return result


def _validate_args(input, target, reduction):
    """Validate input arguments match PyTorch semantics."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input).__name__}")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target must be a Tensor, got {type(target).__name__}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            f"reduction must be 'none', 'sum', or 'mean', got '{reduction}'"
        )
    if input.dim() not in (1, 2):
        raise ValueError(
            f"input must be 1D or 2D, got {input.dim()}D"
        )
    if target.dim() != input.dim():
        raise ValueError(
            f"target dim ({target.dim()}) must match input dim ({input.dim()})"
        )
    if target.dtype != torch.long:
        raise TypeError(
            f"target dtype must be torch.long, got {target.dtype}"
        )
    if target.shape != input.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match input shape {tuple(input.shape)}"
        )
    C = input.shape[-1]
    if C <= 0:
        raise ValueError(f"number of classes must be > 0, got {C}")

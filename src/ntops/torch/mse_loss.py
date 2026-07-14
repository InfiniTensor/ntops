import torch

import ntops
from ntops.torch.utils import _cached_make
_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2
def _get_reduction_enum(reduction):
    if reduction == "none":
        return _REDUCTION_NONE
    if reduction == "mean":
        return _REDUCTION_MEAN
    if reduction == "sum":
        return _REDUCTION_SUM
    raise ValueError("`reduction` must be one of 'none', 'mean', or 'sum'.")
def _as_scalar(output):
    if hasattr(output, "reshape"):
        return output.reshape(())
    if hasattr(output, "view"):
        return output.view(())
    return output
def mse_loss(input, target, reduction="mean"):
    assert input.shape == target.shape, "`input` and `target` must have the same shape."

    reduction_enum = _get_reduction_enum(reduction)

    if reduction_enum == _REDUCTION_NONE:
        output = torch.empty_like(input)
    else:
        output = torch.empty((1,), dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.mse_loss.premake,
        input.ndim,
        reduction_enum,
    )
    if reduction_enum == _REDUCTION_MEAN:
        kernel(input, target, output, float(1.0 / input.numel()))
    else:
        kernel(input, target, output)

    if reduction_enum == _REDUCTION_NONE:
        return output

    return _as_scalar(output)
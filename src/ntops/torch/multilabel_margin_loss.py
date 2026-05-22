import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make


_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2

_MAX_NTOPS_CLASS_SIZE = 32


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


def _is_real_torch_tensor(x):
    return type(x).__module__.startswith("torch")


def _torch_fallback(input, target, reduction):
    class_size = input.shape[-1]

    input_2d = input.reshape(-1, class_size)
    target_2d = target.reshape(-1, class_size)

    output = F.multilabel_margin_loss(
        input_2d,
        target_2d,
        reduction=reduction,
    )

    if reduction == "none":
        return output.reshape(input.shape[:-1])

    return output


def multilabel_margin_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    if size_average is not None or reduce is not None:
        if reduce is False:
            reduction = "none"
        elif size_average is False:
            reduction = "sum"
        else:
            reduction = "mean"

    reduction_enum = _get_reduction_enum(reduction)

    assert input.ndim >= 1, "`input` must have at least 1 dimension."
    assert target.ndim == input.ndim, "`input` and `target` must have the same ndim."
    assert target.shape == input.shape, "`input` and `target` must have the same shape."

    class_size = int(input.shape[-1])

    # 普通 torch.Tensor 可以支持高维 reshape fallback
    # InfiniCore tensor 没有 reshape，所以不能走这个分支
    if _is_real_torch_tensor(input) and input.ndim != 2:
        return _torch_fallback(
            input,
            target,
            reduction,
        )

    assert input.ndim == 2, (
        "`multilabel_margin_loss` ntops path for InfiniCore only supports 2D [N, C]."
    )

    if class_size > _MAX_NTOPS_CLASS_SIZE:
        if _is_real_torch_tensor(input):
            return _torch_fallback(
                input,
                target,
                reduction,
            )

        raise NotImplementedError(
            "class_size is too large for ntops multilabel_margin_loss on InfiniCore tensor."
        )

    per_sample = torch.empty(
        tuple(input.shape[:-1]),
        dtype=input.dtype,
        device=input.device,
    )

    kernel = _cached_make(
        ntops.kernels.multilabel_margin_loss.premake,
        input.ndim,
    )

    kernel(
        input,
        target,
        per_sample,
    )

    if reduction_enum == _REDUCTION_NONE:
        return per_sample

    output = torch.empty(
        (1,),
        dtype=input.dtype,
        device=input.device,
    )

    reduce_kernel = _cached_make(
        ntops.kernels.multilabel_margin_loss.premake_reduce,
        per_sample.ndim,
        reduction_enum,
    )

    if reduction_enum == _REDUCTION_MEAN:
        reduce_kernel(
            per_sample,
            output,
            float(1.0 / per_sample.numel()),
        )
    else:
        reduce_kernel(
            per_sample,
            output,
        )

    return _as_scalar(output)
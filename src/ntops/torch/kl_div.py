import torch

import ntops
from ntops.torch.utils import _cached_make


REDUCTION_SUM = 0
REDUCTION_BATCHMEAN = 1
REDUCTION_MEAN = 2


def _reduction_to_code(reduction):
    if reduction is None:
        reduction = "mean"

    if reduction == "sum":
        return REDUCTION_SUM

    if reduction == "batchmean":
        return REDUCTION_BATCHMEAN

    if reduction == "mean":
        return REDUCTION_MEAN

    raise NotImplementedError(
        "kl_div currently only supports reduction='sum', 'batchmean', or 'mean'"
    )


def kl_div(input, target, reduction="mean", log_target=False):
    assert input.ndim == 2, "kl_div currently only supports 2-D input"
    assert target.shape == input.shape, "target shape must match input shape"
    assert target.dtype == input.dtype, "target dtype must match input dtype"
    assert target.device == input.device, "target device must match input device"

    reduction_code = _reduction_to_code(reduction)
    log_target = bool(log_target)

    # kernel 输出用 shape (1,)，不要用 0-dim。
    # ninetoothed/Triton 对 0-dim output Tensor 容易生成不了 output pointer。
    output = torch.empty((1,), dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.kl_div.premake,
        tuple(input.shape),
        reduction_code,
        log_target,
    )

    kernel(
        input,
        target,
        output,
        input.shape[0],
        input.shape[1],
        reduction_code,
        log_target,
    )

    # 如果是原生 torch.Tensor，可以 reshape 成 PyTorch 一致的 scalar。
    # 如果是 infinicore.Tensor，它没有 reshape，直接返回 (1,)，由 InfiniCore wrapper squeeze。
    if hasattr(output, "reshape"):
        return output.reshape(())

    return output
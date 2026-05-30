import torch

import ntops
from ntops.torch.utils import _cached_make


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def multilabel_margin_loss(input, target, reduction="mean"):
    # One program per sample computes the raw (pre-/C) margin sum; the wrapper
    # divides by C and reduces over N.
    orig_ndim = input.ndim
    if orig_ndim == 1:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    n, c = input.shape
    out = torch.empty((n,), dtype=torch.float32, device=input.device)

    kernel = _cached_make(ntops.kernels.multilabel_margin_loss.premake)
    kernel(input, target.to(torch.int64), out, c=_next_pow2(c))

    out = out / c

    if orig_ndim == 1:
        out = out.squeeze(0)

    if reduction == "mean":
        result = out.mean()
    elif reduction == "sum":
        result = out.sum()
    else:
        result = out

    return result.to(input.dtype)

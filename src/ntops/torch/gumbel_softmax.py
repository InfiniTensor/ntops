import torch

import ntops


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    # Gumbel-Softmax = softmax over `(logits + gumbel_noise) / tau`. The heavy,
    # bandwidth-bound part is the normalized reduction (softmax), which is
    # delegated to the ninetoothed kernel; sampling the noise and (optionally)
    # the straight-through one-hot are cheap torch glue. The noise is drawn
    # identically to `torch.nn.functional.gumbel_softmax` so a shared RNG seed
    # reproduces its result bit-for-bit up to the softmax numerics.
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    gumbels = (logits + gumbels) / tau

    y_soft = ntops.torch.softmax(gumbels, dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        # Straight-through estimator: hard forward value, soft gradient.
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret

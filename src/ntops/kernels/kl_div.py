import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(log_q, log_or_p_target, output, eps, log_target):
    # log_q: always log-probabilities (input to KL divergence)
    # log_or_p_target: probabilities (p) or log-probabilities (log_p) depending on log_target

    if log_target:
        # target is log(p): p = exp(target), log_p = target
        log_p = ntl.cast(log_or_p_target, ntl.float32)
        p = ntl.exp(log_p)
    else:
        # target is p: clip to [eps, 1], then log_p = log(p)
        p = ntl.maximum(
            ntl.cast(log_or_p_target, ntl.float32), ntl.cast(eps, ntl.float32)
        )
        p = ntl.minimum(p, ntl.cast(1.0, ntl.float32))
        log_p = ntl.log(ntl.maximum(p, ntl.cast(eps, ntl.float32)))

    # Clip p for safety, then compute KL loss: p * (log_p - log_q)
    p = ntl.maximum(p, ntl.cast(eps, ntl.float32))
    p = ntl.minimum(p, ntl.cast(1.0, ntl.float32))
    loss = p * (log_p - ntl.cast(log_q, ntl.float32))
    output = ntl.cast(loss, output.dtype)  # noqa: F841


def premake(ndim, eps=1e-10, log_target=False, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),  # log_q (input)
        Tensor(ndim, dtype=dtype),  # log_or_p_target
        Tensor(ndim, dtype=dtype),  # output
        Tensor(0, constexpr=True, value=eps),
        Tensor(0, constexpr=True, value=log_target),
    )

    return arrangement_, application, tensors

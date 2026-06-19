import torch

import ntops
from ntops.torch.utils import _cached_make


def kl_div(input, target, reduction="sum", log_target=False, eps=1e-10):
    """
    Compute the KL divergence loss: p * (log_p - log_q).

    Args:
        input: Log-probabilities (log_q), same shape as target
        target: Probabilities (p) or log-probabilities if log_target=True
        reduction: 'none' | 'sum' | 'mean' | 'batchmean'
        log_target: Whether target is in log space (default: False)
        eps: Epsilon for numerical stability (default: 1e-10)

    Returns:
        KL divergence loss tensor

    Examples:
        >>> log_q = torch.tensor([-0.6931, -0.6931])  # log(0.5)
        >>> p = torch.tensor([0.5, 0.5])
        >>> kl_div(log_q, p, reduction='sum')
        tensor(0.)
    """
    if reduction not in ("none", "sum", "mean", "batchmean"):
        raise ValueError(
            f"reduction must be one of 'none', 'sum', 'mean', 'batchmean', got '{reduction}'"
        )

    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.kl_div.premake, input.ndim, eps, log_target)
    kernel(input, target, output, eps, log_target)

    if reduction == "none":
        return output
    elif reduction == "sum":
        return output.sum()
    elif reduction == "mean":
        return output.mean()
    elif reduction == "batchmean":
        return output.sum() / input.shape[0]

import torch

import ntops
from ntops.torch.utils import _cached_make


def logit(x, eps=1e-6):
    """
    Compute the logit (inverse sigmoid) of a tensor: log(x / (1 - x)).

    Values are clipped to [eps, 1-eps] for numerical stability.

    Args:
        x: Input tensor (should contain values in [0, 1])
        eps: Epsilon for numerical stability (default: 1e-6)

    Returns:
        Tensor with the logit function applied element-wise

    Examples:
        >>> x = torch.tensor([0.1, 0.5, 0.9])
        >>> logit(x)
        tensor([-2.1972,  0.0000,  2.1972])
    """
    output = torch.empty_like(x)

    kernel = _cached_make(ntops.kernels.logit.premake, x.ndim, eps)
    kernel(x, output, eps)

    return output

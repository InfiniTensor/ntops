import math
import random

import torch

import ntops
from ntops.torch.utils import _cached_make


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    if p < 0.0 or p >= 1.0:
        raise ValueError(
            f"dropout probability has to satisfy 0 <= p < 1, but got {p}"
        )

    assert input.ndim >= 2, "Feature dropout requires at least 2 dimensions in the input"

    if not training or p == 0:
        return input

    seed = random.randrange(0, 2**31)

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    alpha_prime = -1.7580993408473766

    q = 1.0 - float(p)
    a = 1.0 / math.sqrt(q * (1.0 + float(p) * alpha_prime * alpha_prime))
    b = -a * float(p) * alpha_prime

    kernel = _cached_make(
        ntops.kernels.feature_alpha_dropout.premake,
        input.ndim,
    )

    kernel(
        input,
        float(p),
        seed,
        float(a),
        float(b),
        output,
    )

    return output
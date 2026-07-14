import torch

import ntops
from ntops.torch.utils import _cached_make


def fliplr(input):
    assert input.ndim >= 2, "`fliplr` requires input with ndim >= 2."

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.flip.premake,
        input.ndim,
        (1,),
    )

    kernel(
        input,
        output,
    )

    return output
import torch
import ntops
from ntops.torch.utils import _cached_make


def hardshrink(input, lambd=0.5):
    if isinstance(lambd, torch.Tensor):
        lambd = float(lambd.item())
    else:
        lambd = float(lambd)

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.hardshrink.hardshrink_premake,
        input.ndim,
        block_size=1024,
    )

    kernel(input, lambd, output)
    return output

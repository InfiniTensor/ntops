import torch

import ntops
from ntops.torch.utils import _cached_make


def where(condition, input, other):
    output = torch.empty_like(input)
    condition_int8 = condition.to(torch.int8)

    kernel = _cached_make(ntops.kernels.where.premake, input.ndim)

    kernel(condition_int8, input, other, output)

    return output

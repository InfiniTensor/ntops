import torch

import ntops
from ntops.torch.utils import _cached_make


def cartesian_prod(*tensors):
    pre_computed = torch.cartesian_prod(*tensors)
    output = torch.empty_like(pre_computed)

    kernel = _cached_make(ntops.kernels.cartesian_prod.premake, pre_computed.ndim)
    kernel(pre_computed, output)

    return output

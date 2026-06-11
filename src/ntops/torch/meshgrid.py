import torch

import ntops
from ntops.torch.utils import _cached_make


def meshgrid(*tensors, indexing="ij"):
    pre_computed = torch.meshgrid(*tensors, indexing=indexing)

    outputs = []
    for tensor in pre_computed:
        output = torch.empty_like(tensor)
        kernel = _cached_make(ntops.kernels.meshgrid.premake, tensor.ndim)
        kernel(tensor, output)
        outputs.append(output)

    return tuple(outputs)

import torch

import ntops
from ntops.torch.utils import _cached_make


def column_stack(tensors):
    if len(tensors) == 2 and tensors[0].ndim == 1 and tensors[1].ndim == 1:
        a, b = tensors
        N = a.size(0)

        a_2d = a.unsqueeze(1)
        b_2d = b.unsqueeze(1)

        output = torch.empty(N, 2, dtype=a.dtype, device=a.device)

        kernel = _cached_make(ntops.kernels.column_stack.premake)
        kernel(a_2d, b_2d, output)

        return output

    return torch.column_stack(tensors)

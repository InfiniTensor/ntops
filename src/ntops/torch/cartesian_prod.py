import torch

import ntops
from ntops.torch.utils import _cached_make


def cartesian_prod(*tensors):
    if len(tensors) == 2:
        a, b = tensors[0], tensors[1]
        n, m = a.size(0), b.size(0)

        a_exp = a.repeat_interleave(m).unsqueeze(1)
        b_exp = b.repeat(n).unsqueeze(1)

        output = torch.empty(n * m, 2, dtype=a.dtype, device=a.device)

        kernel = _cached_make(ntops.kernels.cartesian_prod.premake)
        kernel(a_exp, b_exp, output)

        return output

    return torch.cartesian_prod(*tensors)

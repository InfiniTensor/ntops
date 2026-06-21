import torch

import ntops
from ntops.torch.utils import _cached_make


def channel_shuffle(input, groups):
    n, c, h, w = input.shape

    assert groups > 0
    assert c % groups == 0

    channels_per_group = c // groups

    input = input.view(n, groups, channels_per_group, h, w)
    input = input.transpose(1, 2)

    output = torch.empty((n, c, h, w), dtype=input.dtype, device=input.device)
    kernel = _cached_make(ntops.kernels.channel_shuffle.premake)
    kernel(input, output)

    return output
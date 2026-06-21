import torch

import ntops
from ntops.torch.utils import _cached_make

_ALPHA_PRIME = -1.7580993408473766  # -SELU_SCALE * SELU_ALPHA


def feature_alpha_dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0.0:
        return input if inplace else input.clone()

    if p == 1.0:
        fill = torch.full_like(input, _ALPHA_PRIME)
        if inplace:
            input.copy_(fill)
            return input
        return fill

    # Per-channel Bernoulli mask: shape (N, C, 1, 1, ...), then broadcast and
    # materialize over the spatial dims. Materializing (rather than a stride-0
    # view) is intentional: the MACA/C500 backend handles coalesced contiguous
    # reads far better than stride-0 broadcast loads.
    noise_shape = list(input.shape)
    for i in range(2, input.ndim):
        noise_shape[i] = 1

    noise = torch.empty(noise_shape, dtype=torch.float32, device=input.device)
    noise.bernoulli_(1.0 - p)
    noise = noise.expand_as(input).contiguous()

    output = input if inplace else torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.feature_alpha_dropout.premake, input.ndim)
    kernel(input, noise, 1.0 / (1.0 - p), output)

    return output

import torch

import ntops
from ntops.torch.utils import _cached_make, _pad_dims_to_next_power_of_2


def rot90(input, k=1, dims=(0, 1), *, out=None):
    if out is None:
        if k % 2 == 0:
            out = torch.empty_like(input)
        else:
            dims_permute = list(range(input.ndim))
            dims_permute[dims[0]], dims_permute[dims[1]] = (
                dims_permute[dims[1]],
                dims_permute[dims[0]],
            )
            out = torch.empty(
                input.permute(dims_permute).shape,
                dtype=input.dtype,
                device=input.device,
            )

    if k % 4 == 1:
        input_prepared = _pad_dims_to_next_power_of_2(
            input, dims[1], padding_right=False
        )
    elif k % 4 == 2:
        input_prepared = _pad_dims_to_next_power_of_2(
            input, list(reversed(dims)), padding_right=False
        )
    elif k % 4 == 3:
        input_prepared = _pad_dims_to_next_power_of_2(
            input, dims[0], padding_right=False
        )
    else:
        input_prepared = input

    kernel = _cached_make(ntops.kernels.rot90.premake, input.ndim, k, tuple(dims))

    kernel(input_prepared, out)

    return out

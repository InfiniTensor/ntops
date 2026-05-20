import torch

import ntops
from ntops.torch.utils import _cached_make


def cast(input, dtype):
    output = torch.empty_like(input, dtype=dtype)

    kernel = _cached_make(
        ntops.kernels.cast.premake,
        input.ndim,
        input_dtype=input.dtype,
        output_dtype=dtype,
    )

    kernel(input, output)

    return output

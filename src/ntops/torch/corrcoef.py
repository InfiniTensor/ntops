import torch

import ntops
from ntops.torch.utils import _cached_make


def corrcoef(input):
    assert input.ndim in (1, 2), "corrcoef only supports 1-D or 2-D input"

    if input.ndim == 1:
        output = torch.empty((1,), dtype=input.dtype, device=input.device)
    elif input.shape[0] == 1:
        # torch.corrcoef for shape (1, N) returns scalar.
        output = torch.empty((1,), dtype=input.dtype, device=input.device)
    else:
        output = torch.empty(
            (input.shape[0], input.shape[0]),
            dtype=input.dtype,
            device=input.device,
        )

    kernel = _cached_make(
        ntops.kernels.corrcoef.premake,
        tuple(input.shape),
    )

    if input.ndim == 1:
        kernel(
            input,
            output,
            input.shape[0],
        )

        if hasattr(output, "reshape"):
            return output.reshape(())

        return output

    if input.shape[0] == 1:
        kernel(
            input,
            output,
            input.shape[1],
        )

        if hasattr(output, "reshape"):
            return output.reshape(())

        return output

    kernel(
        input,
        output,
        input.shape[0],
        input.shape[1],
    )

    return output
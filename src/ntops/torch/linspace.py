import torch

import ntops
from ntops.torch.utils import _cached_make


def linspace(start, end, steps, *, dtype=None, device=None, layout=None):
    if device is None:
        device = "cuda"
    if dtype is None:
        dtype = torch.float32

    output = torch.empty(steps, dtype=dtype, device=device)

    compute_dtype = torch.float32 if dtype == torch.float16 else dtype
    start_t = torch.full((steps,), start, dtype=compute_dtype, device=device)
    idx_t = torch.arange(steps, dtype=compute_dtype, device=device)

    if steps > 1:
        step_size = (end - start) / (steps - 1)
    else:
        step_size = 0.0

    step_t = torch.full((steps,), step_size, dtype=compute_dtype, device=device)

    kernel = _cached_make(ntops.kernels.linspace.premake, output.ndim)
    kernel(start_t, idx_t, step_t, output)

    return output

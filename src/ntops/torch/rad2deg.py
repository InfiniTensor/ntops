import torch

import ntops
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


_kernel_1d = None


def _get_kernel_1d():
    global _kernel_1d
    if _kernel_1d is None:
        _kernel_1d = _cached_make(
            ntops.kernels.rad2deg.premake,
            1,
            block_size=ntops.kernels.rad2deg.BLOCK_SIZE,
            num_warps=2,
            max_num_configs=1,
        )
    return _kernel_1d


def _promote_unary_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


def rad2deg(input, *, out=None):
    input = _promote_unary_input(input)
    if input.ndim == 1 and input.is_contiguous():
        if out is None:
            out = torch.empty_like(input)
            _get_kernel_1d()(input, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            _get_kernel_1d()(input, out)
            return out

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)

    kernel_input, kernel_out = _flatten_kernel_tensors(input, out)
    kernel = _cached_make(
        ntops.kernels.rad2deg.premake,
        kernel_input.ndim,
        block_size=ntops.kernels.rad2deg.BLOCK_SIZE,
        num_warps=2,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_out)

    return out

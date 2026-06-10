import torch

import ntops
from ntops.torch.utils import _cached_make


def unbind(input, dim=0):
    if dim < 0:
        dim = input.ndim + dim

    outputs = []

    for i in range(input.shape[dim]):
        slices = [slice(None)] * input.ndim
        slices[dim] = slice(i, i + 1)

        slice_tensor = input[tuple(slices)].squeeze(dim)
        out_slice = torch.empty_like(slice_tensor)

        if slice_tensor.ndim == 0:
            # 0D tensors can't be processed by the kernel.
            out_slice = slice_tensor.clone()
        else:
            kernel = _cached_make(ntops.kernels.unbind.premake, slice_tensor.ndim)
            kernel(slice_tensor, out_slice)

        outputs.append(out_slice)

    return tuple(outputs)

import torch

import ntops
from ntops.torch.utils import _cached_make


def nonzero_sum_gt_last2(input):
    if input.ndim < 2:
        raise AssertionError("nonzero_sum_gt_last2 requires input.ndim >= 2")

    if input.dtype is torch.bool:
        input_for_kernel = input.to(torch.int8)
    else:
        input_for_kernel = input

    output_shape = tuple(input_for_kernel.shape[:-2]) + (1, 1)
    output = torch.empty(
        output_shape, device=input_for_kernel.device, dtype=input_for_kernel.dtype
    )

    kernel = _cached_make(ntops.kernels.nonzero_sum_gt_last2.premake, input.ndim)

    kernel(input_for_kernel, output)

    mask = output.squeeze(-1).squeeze(-1)
    return mask.nonzero()

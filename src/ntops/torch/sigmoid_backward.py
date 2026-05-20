import torch

import ntops
from ntops.torch.utils import _cached_make


def sigmoid_backward(grad_output, output):
    grad_input = torch.empty_like(grad_output)

    kernel = _cached_make(ntops.kernels.sigmoid_backward.premake, grad_output.ndim)

    kernel(grad_output, output, grad_input)

    return grad_input

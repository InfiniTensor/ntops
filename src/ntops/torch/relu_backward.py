import torch

import ntops
from ntops.torch.utils import _cached_make


def relu_backward(grad_output, input):
    grad_input = torch.empty_like(grad_output)

    kernel = _cached_make(ntops.kernels.relu_backward.premake, grad_output.ndim)

    kernel(grad_output, input, grad_input)

    return grad_input

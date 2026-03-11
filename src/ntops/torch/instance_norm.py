import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def instance_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-05,
):
    if weight is None:
        weight = torch.ones(input.shape[1], device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(input.shape[1], device=input.device, dtype=input.dtype)

    tracking_running_stats = False

    if not use_input_stats:
        assert running_mean is not None and running_var is not None, (
            "`running_mean` and `running_var` must be provided when `use_input_stats=False`."
        )
        assert running_mean.shape == (input.shape[1],) and running_var.shape == (
            input.shape[1],
        ), "`running_mean` and `running_var` must have shape (C,)"
    else:
        if running_mean is not None and running_var is not None:
            assert running_mean.shape == (input.shape[1],) and running_var.shape == (
                input.shape[1],
            ), "`running_mean` and `running_var` must have shape (C,)"
            tracking_running_stats = True
            tmp_mean = torch.zeros_like(running_mean)
            tmp_var = torch.zeros_like(running_var)

    output = torch.empty_like(input)

    num_normalized_elements = math.prod(input.shape[2:])
    kernel = _cached_make(
        ntops.kernels.instance_norm.premake,
        input.ndim,
        use_input_stats,
        tracking_running_stats,
        num_normalized_elements,
        block_size=32,
    )

    if use_input_stats:
        if tracking_running_stats:
            kernel(
                input,
                running_mean,
                running_var,
                tmp_mean,
                tmp_var,
                weight,
                bias,
                momentum,
                eps,
                output,
                num_normalized_elements,
            )
        else:
            kernel(
                input,
                weight,
                bias,
                eps,
                output,
                num_normalized_elements,
            )
    else:
        kernel(input, running_mean, running_var, weight, bias, eps, output)

    return output

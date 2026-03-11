import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("eps", (1e-8, 1e-5, 1e-3))
@pytest.mark.parametrize("bias_is_none", (False, True))
@pytest.mark.parametrize("weight_is_none", (False, True))
@pytest.mark.parametrize("use_input_stats", (False, True))
@pytest.mark.parametrize("track_running_stats", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_instance_norm(
    shape,
    dtype,
    device,
    rtol,
    atol,
    weight_is_none,
    bias_is_none,
    use_input_stats,
    track_running_stats,
    eps,
):
    # if len(shape) < 3:
    #     pytest.skip(reason="InstanceNorm requires at least 3D input.")

    while len(shape) < 3:
        shape.insert(0, 1)

    input = torch.randn(shape, dtype=dtype, device=device)

    if weight_is_none:
        weight = None
    else:
        weight = torch.randn(shape[1], dtype=dtype, device=device)

    if bias_is_none:
        bias = None
    else:
        bias = torch.randn(shape[1], dtype=dtype, device=device)

    if use_input_stats and not track_running_stats:
        running_mean = None
        running_var = None
    else:
        running_mean = torch.randn(shape[1], dtype=dtype, device=device)
        running_var = torch.randn(shape[1], dtype=dtype, device=device).abs() + 1

    ninetoothed_output = ntops.torch.instance_norm(
        input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        eps=eps,
    )
    reference_output = torch.nn.functional.instance_norm(
        input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        eps=eps,
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

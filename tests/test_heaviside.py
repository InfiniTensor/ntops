import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_heaviside(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    # Force exact zeros into `input` so the `values` branch is exercised.
    input = torch.where(input > 0.5, torch.zeros_like(input), input)
    values = torch.randn((), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.heaviside(input, values)
    reference_output = torch.heaviside(input, values)

    assert torch.equal(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_heaviside_broadcast_values(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.heaviside(input, values)
    reference_output = torch.heaviside(input, values)

    assert torch.equal(ninetoothed_output, reference_output)

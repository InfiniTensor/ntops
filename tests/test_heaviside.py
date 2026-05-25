

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_heaviside(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn(shape, dtype=dtype, device=device)
    if input.numel() > 0:
        input_flat = input.flatten()
        input_flat[0] = 0

        if input.numel() > 1:
            input_flat[1] = -1

        if input.numel() > 2:
            input_flat[2] = 1

    ninetoothed_output = ntops.torch.heaviside(input, values)
    reference_output = torch.heaviside(input, values)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
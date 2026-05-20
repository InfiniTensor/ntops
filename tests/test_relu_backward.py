import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_relu_backward(shape, dtype, device, rtol, atol):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.relu_backward(grad_output, input)
    reference_output = torch.where(input >= 0, grad_output, torch.zeros_like(grad_output))

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

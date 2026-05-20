import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_tanh_backward(shape, dtype, device, rtol, atol):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.tanh(torch.randn(shape, dtype=dtype, device=device))

    ninetoothed_output = ntops.torch.tanh_backward(grad_output, output)
    reference_output = grad_output * (1 - output * output)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

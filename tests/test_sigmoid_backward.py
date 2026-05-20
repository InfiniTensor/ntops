import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_sigmoid_backward(shape, dtype, device, rtol, atol):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.sigmoid(torch.randn(shape, dtype=dtype, device=device))

    ninetoothed_output = ntops.torch.sigmoid_backward(grad_output, output)
    reference_output = grad_output * output * (1 - output)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

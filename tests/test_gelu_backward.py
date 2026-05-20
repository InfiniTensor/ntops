import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_gelu_backward(shape, dtype, device, rtol, atol):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.gelu_backward(grad_output, input)

    # Use PyTorch autograd for reference
    input_ref = input.clone().requires_grad_(True)
    torch.nn.functional.gelu(input_ref).backward(grad_output)
    reference_output = input_ref.grad

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

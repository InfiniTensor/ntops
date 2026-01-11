import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_logical_and(shape, dtype, device, rtol, atol):
    input = torch.randint(0, 2, shape, device=device).to(dtype)
    other = torch.randint(0, 2, shape, device=device).to(dtype)

    ninetoothed_output = ntops.torch.logical_and(input, other)
    reference_output = torch.logical_and(input, other).to(dtype)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
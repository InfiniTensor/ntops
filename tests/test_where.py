import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_where(shape, dtype, device, rtol, atol):
    condition = torch.rand(shape, device=device) > 0.5
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.where(condition, input, other)
    reference_output = torch.where(condition, input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

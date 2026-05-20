import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cast(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.cast(input, torch.float32)
    reference_output = input.to(torch.float32)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

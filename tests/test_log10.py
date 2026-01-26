import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_log10(shape, dtype, device, rtol, atol):
    input = torch.rand(shape, dtype=dtype, device=device) + 0.1

    ninetoothed_output = ntops.torch.log10(input)
    reference_output = torch.log10(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

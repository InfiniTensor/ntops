import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_lgamma(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device).abs() + 0.1

    ninetoothed_output = ntops.torch.lgamma(input)
    reference_output = torch.lgamma(input)

    assert torch.allclose(
        ninetoothed_output, reference_output, rtol=rtol, atol=atol, equal_nan=True
    )

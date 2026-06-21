import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_frac(shape, dtype, device, rtol, atol):
    # Spread values so integer parts are nontrivial (exercises trunc).
    input = torch.randn(shape, dtype=dtype, device=device) * 5

    ninetoothed_output = ntops.torch.frac(input)
    reference_output = torch.frac(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

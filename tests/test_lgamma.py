import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_lgamma(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return
    # Use positive values since lgamma is defined for positive inputs.
    input = torch.rand(shape, dtype=dtype, device=device) * 5 + 0.1

    ninetoothed_output = ntops.torch.lgamma(input)
    reference_output = torch.lgamma(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

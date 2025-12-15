import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_log2(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return
    # Use positive values to avoid log of negative numbers
    input = torch.abs(torch.randn(shape, dtype=dtype, device=device)) + 1e-6

    ninetoothed_output = ntops.torch.log2(input)
    reference_output = torch.log2(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

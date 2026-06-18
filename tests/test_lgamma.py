import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_lgamma_float(shape, dtype, device, rtol, atol):
    input = torch.rand(shape, dtype=dtype, device=device) * 5 + 0.1

    ninetoothed_output = ntops.torch.lgamma(input)
    reference_output = torch.lgamma(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments(False))
def test_lgamma_int(shape, dtype, device, rtol, atol):
    # torch.lgamma on integers returns float32
    input = torch.randint(1, 20, size=shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.lgamma(input)
    reference_output = torch.lgamma(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=1e-4, atol=1e-4)

import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_moveaxis_single_dim(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    if input.ndim == 0:
        pytest.skip("moveaxis does not support scalar input")

    source = random.randint(0, input.ndim - 1)
    destination = random.randint(0, input.ndim - 1)

    ninetoothed_output = ntops.torch.moveaxis(input, source, destination)
    reference_output = torch.moveaxis(input, source, destination)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
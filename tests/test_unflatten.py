import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_unflatten(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    if input.ndim == 0:
        pytest.skip("unflatten does not support scalar input")

    dim = random.randint(0, input.ndim - 1)
    dim_size = input.shape[dim]

    sizes = (1, dim_size)

    ninetoothed_output = ntops.torch.unflatten(input, dim, sizes)
    reference_output = torch.unflatten(input, dim, sizes)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
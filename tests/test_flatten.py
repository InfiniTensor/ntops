import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_flatten_default(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.flatten(input)
    reference_output = torch.flatten(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
    assert ninetoothed_output.shape == reference_output.shape


@skip_if_cuda_not_available
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_flatten_partial(ndim):
    shape = tuple(range(2, ndim + 2))
    input = torch.randn(shape, device="cuda")

    for start_dim in range(ndim):
        for end_dim in range(start_dim, ndim):
            ninetoothed_output = ntops.torch.flatten(input, start_dim, end_dim)
            reference_output = torch.flatten(input, start_dim, end_dim)

            assert torch.allclose(ninetoothed_output, reference_output)
            assert ninetoothed_output.shape == reference_output.shape

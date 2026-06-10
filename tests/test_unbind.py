import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_unbind(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.unbind(input)
    reference_output = torch.unbind(input)

    assert len(ninetoothed_output) == len(reference_output)

    for ninetoothed_tensor, reference_tensor in zip(ninetoothed_output, reference_output):
        assert torch.allclose(ninetoothed_tensor, reference_tensor, rtol=rtol, atol=atol)
        assert ninetoothed_tensor.shape == reference_tensor.shape


@skip_if_cuda_not_available
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_unbind_dims(ndim):
    shape = tuple(range(2, ndim + 2))
    input = torch.randn(shape, device="cuda")

    for dim in range(ndim):
        ninetoothed_output = ntops.torch.unbind(input, dim)
        reference_output = torch.unbind(input, dim)

        assert len(ninetoothed_output) == len(reference_output)

        for ninetoothed_tensor, reference_tensor in zip(ninetoothed_output, reference_output):
            assert torch.allclose(ninetoothed_tensor, reference_tensor)
            assert ninetoothed_tensor.shape == reference_tensor.shape


@skip_if_cuda_not_available
def test_unbind_concatenation():
    input = torch.randn(3, 4, 5, device="cuda")

    for dim in range(3):
        ninetoothed_output = ntops.torch.unbind(input, dim)
        stacked = torch.stack(ninetoothed_output, dim)

        assert torch.allclose(stacked, input)

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_repeat_same(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return

    input = torch.randn(shape, dtype=dtype, device=device)
    repeats = tuple(2 for _ in range(input.ndim))

    ninetoothed_output = ntops.torch.repeat(input, *repeats)
    reference_output = input.repeat(*repeats)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
    assert ninetoothed_output.shape == reference_output.shape


@skip_if_cuda_not_available
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_repeat_various(ndim):
    shape = tuple(range(2, ndim + 2))
    input = torch.randn(shape, device="cuda")

    repeat_specs = [
        tuple(1 for _ in range(ndim)),
        tuple(3 for _ in range(ndim)),
        (1,) * (ndim - 1) + (4,),
        (2, 3) if ndim >= 2 else (1,),
    ]

    for repeats in repeat_specs:
        if len(repeats) != ndim:
            continue

        ninetoothed_output = ntops.torch.repeat(input, *repeats)
        reference_output = input.repeat(*repeats)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == reference_output.shape


@skip_if_cuda_not_available
def test_repeat_list_input():
    input = torch.tensor([[1, 2], [3, 4]], device="cuda")
    sizes = (2, 3)

    # Test with *sizes unpacking
    ninetoothed_output = ntops.torch.repeat(input, *sizes)
    reference_output = input.repeat(*sizes)

    assert torch.equal(ninetoothed_output, reference_output)

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def _int_nextafter_ref(input, other):
    # torch.nextafter doesn't support integers; step ±1 toward other.
    result = input.clone()
    result[input < other] += 1
    result[input > other] -= 1
    return result


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_nextafter_float(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.nextafter(input, other)
    reference_output = torch.nextafter(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments(False))
def test_nextafter_int(shape, dtype, device, rtol, atol):
    input = torch.randint(-100, 100, size=shape, dtype=dtype, device=device)
    other = torch.randint(-100, 100, size=shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.nextafter(input, other)
    reference_output = _int_nextafter_ref(input, other)

    assert torch.equal(ninetoothed_output, reference_output)

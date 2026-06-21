import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_narrow(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    dim = input.ndim - 1
    size = input.shape[dim]

    start = size // 3
    length = size - start

    ninetoothed_output = ntops.torch.narrow(input, dim, start, length)
    reference_output = torch.narrow(input, dim, start, length)

    assert torch.equal(ninetoothed_output, reference_output)
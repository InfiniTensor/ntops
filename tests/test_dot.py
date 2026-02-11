import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("length", [1024, 8192])
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float16, 0.01, 0.01),
        (torch.float32, 0.001, 0.001),
    ],
)
def test_dot(length, dtype, rtol, atol):
    device = "cuda"
    input = torch.randn((length,), device=device, dtype=dtype)
    other = torch.randn((length,), device=device, dtype=dtype)

    ninetoothed_output = ntops.torch.dot(input, other)
    reference_output = torch.dot(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

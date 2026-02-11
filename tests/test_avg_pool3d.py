import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, ceil_mode",
    [
        ((3, 4, 5, 6), 2, 2, False),
        ((2, 3, 5, 6, 7), (3, 2, 2), None, True),
    ],
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float16, 0.01, 0.01),
        (torch.float32, 0.001, 0.001),
    ],
)
def test_avg_pool3d(input_shape, kernel_size, stride, ceil_mode, dtype, rtol, atol):
    device = "cuda"
    input = torch.randn(input_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.avg_pool3d(
        input, kernel_size, stride=stride, ceil_mode=ceil_mode
    )

    reference_input = input if input.ndim == 5 else input.unsqueeze(0)
    reference_output = torch.nn.functional.avg_pool3d(
        reference_input, kernel_size, stride=stride, ceil_mode=ceil_mode
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

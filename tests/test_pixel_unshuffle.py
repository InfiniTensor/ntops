import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("downscale_factor", (1, 2, 4))
@pytest.mark.parametrize(
    "input_shape",
    (
        (13, 1, 8, 8),
        (2, 3, 16, 24),
        (1, 4, 32, 32),
    ),
)
def test_pixel_unshuffle(shape, dtype, device, rtol, atol, downscale_factor, input_shape):
    del shape

    input = torch.randn(input_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.pixel_unshuffle(
        input,
        downscale_factor,
    )

    reference_output = F.pixel_unshuffle(
        input,
        downscale_factor,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )
import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
    ),
)
@pytest.mark.parametrize("groups", (1, 2, 3, 4, 6))
@pytest.mark.parametrize("n, c, h, w", ((2, 12, 32, 32),))
def test_channel_shuffle(n, c, h, w, groups, dtype, device, rtol, atol):
    input = torch.randn((n, c, h, w), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.channel_shuffle(input, groups)
    reference_output = F.channel_shuffle(input, groups)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
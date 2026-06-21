import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

_CASES = [
    # (N, C, H, W, kH, kW, oH, oW)
    (1, 1, 16, 16, 2, 2, 8, 8),
    (2, 3, 16, 16, 2, 2, 12, 12),
    (2, 3, 32, 32, 3, 3, 20, 20),
    (1, 4, 24, 24, 2, 2, 16, 16),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("n, c, h, w, kh, kw, oh, ow", _CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_fractional_max_pool2d(n, c, h, w, kh, kw, oh, ow, dtype):
    device = "cuda"
    input = torch.randn(n, c, h, w, dtype=dtype, device=device)
    samples = torch.rand(n, c, 2, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.fractional_max_pool2d(
        input, (kh, kw), output_size=(oh, ow), _random_samples=samples
    )
    reference_output = F.fractional_max_pool2d(
        input, (kh, kw), output_size=(oh, ow), _random_samples=samples
    )

    assert torch.equal(ninetoothed_output, reference_output)

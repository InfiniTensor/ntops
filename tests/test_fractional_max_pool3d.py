import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

_CASES = [
    # (N, C, D, H, W, kD, kH, kW, oD, oH, oW)
    (1, 1, 8, 8, 8, 2, 2, 2, 4, 4, 4),
    (2, 3, 8, 12, 16, 2, 2, 2, 6, 9, 12),
    (1, 2, 12, 12, 12, 3, 3, 3, 8, 8, 8),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("n, c, d, h, w, kd, kh, kw, od, oh, ow", _CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_fractional_max_pool3d(n, c, d, h, w, kd, kh, kw, od, oh, ow, dtype):
    device = "cuda"
    input = torch.randn(n, c, d, h, w, dtype=dtype, device=device)
    samples = torch.rand(n, c, 3, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.fractional_max_pool3d(
        input, (kd, kh, kw), output_size=(od, oh, ow), _random_samples=samples
    )
    reference_output = F.fractional_max_pool3d(
        input, (kd, kh, kw), output_size=(od, oh, ow), _random_samples=samples
    )

    assert torch.equal(ninetoothed_output, reference_output)

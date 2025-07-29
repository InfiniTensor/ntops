import pytest
import torch
import torch.nn.functional as F

import ntops.torch
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype, atol, rtol", ((torch.float16, 0.025, 0.025), (torch.float32, 0.01, 0.01))
)
@pytest.mark.parametrize("dilation", (1, 2, (2, 3)))
@pytest.mark.parametrize("stride", (1, 2, (2, 3)))
@pytest.mark.parametrize("r, s", ((1, 1), (3, 3)))
@pytest.mark.parametrize("n, c, h, w, k", ((2, 3, 112, 112, 4),))
def test_cuda(n, c, h, w, k, r, s, stride, dilation, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((k, c, r, s), dtype=dtype, device=device)
    bias = torch.randn((k,), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.conv2d(
        input, weight, bias=bias, stride=stride, dilation=dilation
    )
    reference_output = F.conv2d(
        input, weight, bias=bias, stride=stride, dilation=dilation
    )

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)

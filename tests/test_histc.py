import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("bins", [10, 50])
@pytest.mark.parametrize("is_moore", [False, True])
def test_histc(bins, is_moore):
    dtype = torch.float32
    device = "cuda"
    input = torch.randn((2048,), dtype=dtype, device=device) * 5 - 2
    min_val, max_val = -5.0, 5.0

    ninetoothed_output = ntops.torch.histc(
        input, bins=bins, min=min_val, max=max_val, is_moore=is_moore
    )
    reference_output = torch.histc(input, bins=bins, min=min_val, max=max_val)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=0, atol=1e-3)

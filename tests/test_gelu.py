import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_gelu(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    for approximate in ("none", "tanh"):
        ninetoothed_output = ntops.torch.gelu(input)
        reference_output = F.gelu(input)

        assert torch.allclose(
            ninetoothed_output, reference_output, rtol=rtol, atol=atol
        )

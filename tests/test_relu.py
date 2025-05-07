import pytest
import torch
import torch.nn.functional as F

import ntops.torch
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(shape, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)

    for replace in (False, True):
        ninetoothed_output = ntops.torch.relu(input, replace)
        reference_output = F.relu(input, replace)

        assert torch.allclose(
            ninetoothed_output, reference_output, atol=atol, rtol=rtol
        )

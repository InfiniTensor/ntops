import pytest
import torch

import ntops.torch
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(shape, dtype, atol, rtol):
    device = "cuda"

    def generate_inf_tensor(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=device)
        inf_prob = 0.5
        mask = torch.rand(shape) < inf_prob
        x[mask] = float("inf")
        x[~mask] = float("-inf")
        return x

    input = generate_inf_tensor(shape, dtype, device)

    ninetoothed_output = ntops.torch.isinf(input)
    reference_output = torch.isinf(input)

    assert torch.equal(ninetoothed_output, reference_output)

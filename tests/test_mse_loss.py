import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_mse_loss(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)

    reduction = random.choice(("none", "mean", "sum"))

    ninetoothed_output = ntops.torch.mse_loss(
        input,
        target,
        reduction=reduction,
    )

    reference_output = F.mse_loss(
        input,
        target,
        reduction=reduction,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )
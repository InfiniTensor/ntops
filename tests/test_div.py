import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_div(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    for rounding_mode in (None, "trunc", "floor"):
        # TODO: Test for `trunc` mode later.
        if rounding_mode == "trunc":
            continue

        ninetoothed_output = ntops.torch.div(input, other, rounding_mode=rounding_mode)
        reference_output = torch.div(input, other, rounding_mode=rounding_mode)

        assert torch.allclose(
            ninetoothed_output, reference_output, rtol=rtol, atol=atol
        )

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments(False))
def test_lcm(shape, dtype, device, rtol, atol):
    if dtype == torch.bool:
        input = torch.randint(1, 10, size=shape, dtype=torch.int32, device=device)
        other = torch.randint(1, 10, size=shape, dtype=torch.int32, device=device)
    else:
        upper_bound = 20
        input = torch.randint(
            1, upper_bound, size=shape, dtype=dtype, device=device
        )
        other = torch.randint(
            1, upper_bound, size=shape, dtype=dtype, device=device
        )

    ninetoothed_output = ntops.torch.lcm(input, other)
    reference_output = torch.lcm(input, other)

    assert torch.equal(ninetoothed_output, reference_output)

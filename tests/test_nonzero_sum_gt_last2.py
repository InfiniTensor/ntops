import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _make_mask(shape, dtype, device):
    base = torch.randint(0, 2, shape, device=device, dtype=torch.int32)
    return base.to(dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(2, 3), (2, 5, 7), (3, 4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32, torch.bool])
def test_nonzero_sum_gt_last2(shape, dtype):
    device = "cuda"
    input = _make_mask(shape, dtype, device)

    ninetoothed_output = ntops.torch.nonzero_sum_gt_last2(input)
    reference_output = torch.greater(input.sum(dim=(-1, -2)), 0).nonzero()

    assert torch.equal(ninetoothed_output, reference_output)

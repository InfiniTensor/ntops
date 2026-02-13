import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 2, 3), (2, 1, 3, 4)])
@pytest.mark.parametrize("num_classes", [-1, 5, 16])
def test_one_hot(shape, num_classes):
    device = "cuda"
    dtype = torch.int64

    if num_classes == -1:
        max_class = 7
        input = torch.randint(0, max_class, size=shape, device=device, dtype=dtype)
        ninetoothed_output = ntops.torch.one_hot(input)
        reference_output = torch.nn.functional.one_hot(input)
    else:
        input = torch.randint(0, num_classes, size=shape, device=device, dtype=dtype)
        ninetoothed_output = ntops.torch.one_hot(input, num_classes)
        reference_output = torch.nn.functional.one_hot(input, num_classes)

    assert torch.equal(ninetoothed_output, reference_output)

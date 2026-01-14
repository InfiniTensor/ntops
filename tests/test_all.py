import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("keepdim", [False, True])
def test_all(shape, dtype, device, rtol, atol, keepdim):
    if dtype == torch.bool:
        input_tensor = torch.randint(0, 2, shape, device=device).bool()
    else:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
        mask = torch.rand(shape, device=device) < 0.2
        input_tensor[mask] = 0

    if random.random() < 0.2:
        dim = None
    else:
        dim = random.randint(0, len(shape) - 1)
        if random.choice([True, False]):
            dim -= len(shape)

    ntops_res = ntops.torch.all(input_tensor, dim=dim, keepdim=keepdim)

    if dim is None:
        ref_res = torch.all(input_tensor)
    else:
        ref_res = torch.all(input_tensor, dim=dim, keepdim=keepdim)

    assert torch.equal(ntops_res, ref_res)

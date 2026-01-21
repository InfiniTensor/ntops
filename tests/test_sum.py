import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("keepdim", (False, True))
def test_sum_dim(shape, dtype, device, rtol, atol, keepdim):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1)

    if random.choice((True, False)):
        dim = dim - input_tensor.ndim

    ntops_value = ntops.torch.sum(input_tensor, dim=dim, keepdim=keepdim)
    reference_value = torch.sum(input_tensor, dim=dim, keepdim=keepdim)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_sum_global(shape, dtype, device, rtol, atol):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_value = ntops.torch.sum(input_tensor)
    reference_value = torch.sum(input_tensor)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)

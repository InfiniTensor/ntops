import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def _make_unique_along_dim(input, dim, dtype):
    dim_size = input.shape[dim]
    view_shape = [1] * input.ndim
    view_shape[dim] = dim_size
    offset = torch.arange(dim_size, device=input.device, dtype=dtype).view(view_shape)
    epsilon = 0.01 if dtype == torch.float16 else 1e-4
    return input + offset * epsilon


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_topk(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    dim = -1
    dim_size = input.shape[dim]
    k = random.randint(1, min(dim_size, 8))

    input = _make_unique_along_dim(input, dim, dtype)

    ninetoothed_values, ninetoothed_indices = ntops.torch.topk(input, k, dim=dim)
    reference_values, reference_indices = torch.topk(
        input, k, dim=dim, largest=True, sorted=True
    )

    assert torch.allclose(ninetoothed_values, reference_values, rtol=rtol, atol=atol)
    assert torch.equal(ninetoothed_indices, reference_indices)

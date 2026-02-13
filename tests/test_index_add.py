import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import gauss, generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_index_add(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    dim = 0
    dim_size = input.shape[dim]
    index_size = random.randint(1, dim_size)

    index = torch.randint(0, dim_size, (index_size,), device=device, dtype=torch.int64)

    source_shape = list(shape)
    source_shape[dim] = index_size
    source = torch.randn(source_shape, dtype=dtype, device=device)

    alpha = gauss()

    ninetoothed_output = ntops.torch.index_add(input, dim, index, source, alpha=alpha)
    reference_output = torch.index_add(input, dim, index, source, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

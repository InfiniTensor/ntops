import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_var_dim(shape, dtype, device, rtol, atol, correction, keepdim):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1)

    if random.choice((True, False)):
        dim = dim - input_tensor.ndim

    ntops_value = ntops.torch.var(
        input_tensor, dim=dim, correction=correction, keepdim=keepdim
    )

    reference_value = torch.var(
        input_tensor, dim=dim, correction=correction, keepdim=keepdim
    )

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("correction", [0, 1])
def test_var_global(shape, dtype, device, rtol, atol, correction):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_value = ntops.torch.var(input_tensor, correction=correction)
    reference_value = torch.var(input_tensor, correction=correction)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)

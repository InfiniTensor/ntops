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
def test_var_mean_general(keepdim, correction, shape, dtype, device, rtol, atol):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1) if input_tensor.ndim > 0 else None
    if dim is not None and random.choice((True, False)):
        dim = dim - input_tensor.ndim

    nt_var, nt_mean = ntops.torch.var_mean(
        input_tensor, dim=dim, correction=correction, keepdim=keepdim
    )
    ref_var, ref_mean = torch.var_mean(
        input_tensor, dim=dim, correction=correction, keepdim=keepdim
    )

    assert torch.allclose(nt_var, ref_var, rtol=rtol, atol=atol, equal_nan=True)
    assert torch.allclose(nt_mean, ref_mean, rtol=rtol, atol=atol, equal_nan=True)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_var_mean_global(shape, dtype, device, rtol, atol):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    nt_var, nt_mean = ntops.torch.var_mean(input_tensor, dim=None)
    ref_var, ref_mean = torch.var_mean(input_tensor, dim=None)

    assert torch.allclose(nt_var, ref_var, rtol=rtol, atol=atol, equal_nan=True)
    assert torch.allclose(nt_mean, ref_mean, rtol=rtol, atol=atol, equal_nan=True)

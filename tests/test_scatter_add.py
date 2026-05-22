import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_scatter_add(shape, dtype, device, rtol, atol):
    input = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    src = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    dim = random.randrange(len(shape))

    if shape[dim] == 0:
        pytest.skip("scatter_add dim size must be non-zero.")

    index = torch.randint(
        0,
        shape[dim],
        shape,
        dtype=torch.long,
        device=device,
    )

    ninetoothed_output = ntops.torch.scatter_add(
        input,
        dim,
        index,
        src,
    )

    reference_output = torch.scatter_add(
        input,
        dim,
        index,
        src,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )
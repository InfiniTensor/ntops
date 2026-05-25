import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_slice_scatter(shape, dtype, device, rtol, atol):
    if len(shape) == 0:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    dim = input.ndim - 1
    size = input.shape[dim]

    start = size // 3
    end = size
    step = 1

    source_shape = list(shape)
    source_shape[dim] = end - start

    source = torch.randn(source_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.slice_scatter(
        input,
        source,
        dim=dim,
        start=start,
        end=end,
        step=step,
    )

    reference_output = torch.slice_scatter(
        input,
        source,
        dim=dim,
        start=start,
        end=end,
        step=step,
    )

    assert torch.equal(ninetoothed_output, reference_output)
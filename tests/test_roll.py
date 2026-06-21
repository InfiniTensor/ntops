import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize(
    "shape, shifts, dims",
    [
        ((10,), 3, None),
        ((10,), -2, None),
        ((3, 5), 1, 0),
        ((3, 5), (1, 2), (0, 1)),
        ((4, 6, 8), (2, -1), (1, 2)),
        ((5,), 7, 0),
    ],
)
def test_roll(shape, shifts, dims, dtype):
    input = torch.arange(
        shape[0] if len(shape) == 1 else 1,
        dtype=dtype,
        device="cuda",
    )
    if len(shape) > 1:
        input = input.unsqueeze(-1).expand(shape).contiguous().clone()

    ninetoothed_output = ntops.torch.roll(input, shifts, dims)
    reference_output = torch.roll(input, shifts, dims)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.equal(ninetoothed_output, reference_output)

import pytest
import torch

import ntops
from tests.utils import generate_arguments


_COUNT_NONZERO_TEST_CASES = [
    ((8, 8), None, None),
    ((8, 8), (16, 1), 1),
    ((2, 3, 4), None, 0),
    ((1, 8), None, (0,)),
    ((16, 64), (128, 1), None),
    ((4, 5, 6), (60, 12, 2), 2),
]

_SUPPORTED_DTYPES = (
    torch.int32,
    torch.float32,
    torch.uint8,
)


@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("input_shape,strides,dim", _COUNT_NONZERO_TEST_CASES)
def test_count_nonzero(shape, dtype, device, rtol, atol, input_shape, strides, dim):
    _ = (shape, rtol, atol)

    if dtype not in _SUPPORTED_DTYPES:
        return

    if dtype == torch.uint8:
        base = torch.randint(
            0,
            3,
            input_shape,
            dtype=dtype,
            device=device,
        )
    else:
        base = torch.randint(
            -2,
            3,
            input_shape,
            device=device,
        ).to(dtype)

    if strides is not None:
        input = torch.empty_strided(
            input_shape,
            strides,
            dtype=dtype,
            device=device,
        )
        input.copy_(base)
    else:
        input = base

    if dim is None:
        output = ntops.torch.count_nonzero(input)
        reference = torch.count_nonzero(input)
    else:
        output = ntops.torch.count_nonzero(input, dim=dim)
        reference = torch.count_nonzero(input, dim=dim)

    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert output.device == reference.device
    assert torch.equal(output, reference)
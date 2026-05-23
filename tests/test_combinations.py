import pytest
import torch

import ntops
from tests.utils import generate_arguments


_COMBINATIONS_TEST_CASES = [
    (5, 1, False),
    (5, 2, False),
    (6, 3, False),
    (7, 2, False),
    (8, 3, False),
    (3, 2, False),
]


@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("size,r,with_replacement", _COMBINATIONS_TEST_CASES)
def test_combinations(shape, dtype, device, rtol, atol, size, r, with_replacement):
    _ = (shape, rtol, atol)

    if dtype != torch.float32:
        return

    input = torch.randn(
        (size,),
        dtype=dtype,
        device=device,
    )

    output = ntops.torch.combinations(
        input,
        r=r,
        with_replacement=with_replacement,
    )

    reference = torch.combinations(
        input,
        r=r,
        with_replacement=with_replacement,
    )

    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert output.device == reference.device

    assert torch.equal(output, reference)
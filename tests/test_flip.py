import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize(
    "input_shape,dims,input_strides",
    (
        ((13, 4), (0,), None),
        ((8, 16), (1,), (128, 1)),
        ((2, 3, 4), (2,), None),
        ((4, 5, 6), (0, 2), None),
        ((16, 64), (0, 1), None),
        ((2, 2, 3, 4), (1, 3), None),
    ),
)
def test_flip(shape, dtype, device, rtol, atol, input_shape, dims, input_strides):
    del shape

    if input_strides is None:
        input = torch.randn(input_shape, dtype=dtype, device=device)
    else:
        storage_size = 1
        for size, stride in zip(input_shape, input_strides):
            storage_size += (size - 1) * stride

        base = torch.randn((storage_size,), dtype=dtype, device=device)
        input = torch.as_strided(
            base,
            size=input_shape,
            stride=input_strides,
        )

    ninetoothed_output = ntops.torch.flip(
        input,
        dims,
    )

    reference_output = torch.flip(
        input,
        dims,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )
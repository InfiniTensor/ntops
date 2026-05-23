import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


_CORRCOEF_TEST_CASES = [
    ((5,), None),
    ((3, 5), None),
    ((4, 4), None),
    ((2, 8), None),
    ((6, 6), None),
    ((1, 7), None),
]


def _generate_float32_arguments():
    arg_names, arg_values = generate_arguments()

    filtered_arg_values = [
        args for args in arg_values
        if args[1] == torch.float32
    ]

    return arg_names, filtered_arg_values


@skip_if_cuda_not_available
@pytest.mark.parametrize(*_generate_float32_arguments())
@pytest.mark.parametrize("case_shape,case_strides", _CORRCOEF_TEST_CASES)
def test_corrcoef(shape, dtype, device, rtol, atol, case_shape, case_strides):
    rtol, atol = 5e-4, 5e-4

    base = torch.randn(case_shape, dtype=dtype, device=device)

    if case_strides is not None:
        input = torch.empty_strided(
            case_shape,
            case_strides,
            dtype=dtype,
            device=device,
        )
        input.copy_(base)
    else:
        input = base

    ninetoothed_output = ntops.torch.corrcoef(input)
    reference_output = torch.corrcoef(input)

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
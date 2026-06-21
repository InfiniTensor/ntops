import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments as generate_base_arguments


def generate_arguments():
    names = "shape,strides,dtype,device,rtol,atol"

    _, base_values = generate_base_arguments()

    matrix_cases = [
        ((1, 1), None),
        ((2, 2), None),
        ((3, 3), (3, 1)),
        ((4, 4), None),
        ((8, 8), (512, 1)),
        ((16, 16), None),
    ]

    values = []

    for _, dtype, device, rtol, atol in base_values:
        if dtype != torch.float32:
            continue

        for shape, strides in matrix_cases:
            values.append(
                (
                    shape,
                    strides,
                    dtype,
                    device,
                    rtol,
                    atol,
                )
            )

    return names, values


def _make_input(shape, strides, dtype, device):
    if strides is None:
        input = torch.randn(shape, dtype=dtype, device=device)
    else:
        input = torch.empty_strided(shape, strides, dtype=dtype, device=device)
        input.normal_()
    input.diagonal().add_(0.1)

    return input


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_slogdet(shape, strides, dtype, device, rtol, atol):
    input = _make_input(shape, strides, dtype, device)

    ninetoothed_sign, ninetoothed_logabsdet = ntops.torch.slogdet(input)
    reference_sign, reference_logabsdet = torch.slogdet(input)

    assert ninetoothed_sign.shape == reference_sign.shape
    assert ninetoothed_logabsdet.shape == reference_logabsdet.shape

    assert ninetoothed_sign.dtype == reference_sign.dtype
    assert ninetoothed_logabsdet.dtype == reference_logabsdet.dtype

    assert ninetoothed_sign.device == reference_sign.device
    assert ninetoothed_logabsdet.device == reference_logabsdet.device

    assert torch.allclose(
        ninetoothed_sign,
        reference_sign,
        rtol=rtol,
        atol=atol,
    )

    assert torch.allclose(
        ninetoothed_logabsdet,
        reference_logabsdet,
        rtol=rtol,
        atol=atol,
    )
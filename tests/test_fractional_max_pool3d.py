import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


_TEST_CASES_DATA = [
    ((2, 3, 9, 9, 9), None, (3, 3, 3), (4, 4, 4), False),
    ((1, 4, 8, 10, 12), None, (2, 3, 2), (4, 4, 6), False),
    ((2, 2, 7, 11, 5), (770, 110, 55, 5, 1), (3, 2, 3), (3, 4, 2), False),
    ((3, 6, 5, 6, 7), None, (2, 2, 2), (3, 3, 4), False),
    ((1, 8, 10, 10, 10), None, (4, 3, 2), (5, 4, 5), False),
    ((2, 5, 12, 8, 6), None, (3, 3, 2), (4, 3, 2), False),
]


_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
}


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.empty(0).cuda().device


def _get_tolerance(dtype):
    return _TOLERANCE_MAP[dtype]


def _make_input(shape, strides, dtype, device):
    if strides is None:
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
        )

    storage_size = 1
    for size, stride in zip(shape, strides):
        storage_size += (size - 1) * stride

    base = torch.randn(
        (storage_size,),
        dtype=dtype,
        device=device,
    )

    return torch.as_strided(
        base,
        size=shape,
        stride=strides,
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize(
    "in_shape, in_strides, kernel_size, output_size, return_indices",
    _TEST_CASES_DATA,
)
def test_fractional_max_pool3d_cases(
    in_shape,
    in_strides,
    kernel_size,
    output_size,
    return_indices,
    dtype,
    device,
):
    assert not return_indices, "return_indices is not supported by ntops yet."

    input = _make_input(
        in_shape,
        in_strides,
        dtype=dtype,
        device=device,
    )

    n, c, d, h, w = in_shape

    random_samples = torch.rand(
        (n, c, 3),
        dtype=dtype,
        device=input.device,
    )

    ninetoothed_output = ntops.torch.fractional_max_pool3d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=None,
        return_indices=return_indices,
        _random_samples=random_samples,
    )

    reference_output = F.fractional_max_pool3d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=None,
        return_indices=return_indices,
        _random_samples=random_samples,
    )

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype

    assert torch.isfinite(ninetoothed_output).all(), (
        "ninetoothed_output contains inf or nan"
    )
    assert torch.isfinite(reference_output).all(), (
        "reference_output contains inf or nan"
    )

    tolerance = _get_tolerance(dtype)

    torch.testing.assert_close(
        ninetoothed_output,
        reference_output,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("kernel_size", (2, (3, 3, 3)))
@pytest.mark.parametrize(
    "output_size, output_ratio",
    (
        ((6, 6, 6), None),
        (None, (0.5, 0.5, 0.5)),
    ),
)
@pytest.mark.parametrize("n, c, d, h, w", ((2, 3, 12, 12, 12),))
def test_fractional_max_pool3d_ratio_and_size(
    n,
    c,
    d,
    h,
    w,
    kernel_size,
    output_size,
    output_ratio,
    dtype,
    device,
):
    input = torch.randn(
        (n, c, d, h, w),
        dtype=dtype,
        device=device,
    )

    random_samples = torch.rand(
        (n, c, 3),
        dtype=dtype,
        device=input.device,
    )

    ninetoothed_output = ntops.torch.fractional_max_pool3d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=False,
        _random_samples=random_samples,
    )

    reference_output = F.fractional_max_pool3d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=False,
        _random_samples=random_samples,
    )

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype

    assert torch.isfinite(ninetoothed_output).all(), (
        "ninetoothed_output contains inf or nan"
    )
    assert torch.isfinite(reference_output).all(), (
        "reference_output contains inf or nan"
    )

    tolerance = _get_tolerance(dtype)

    torch.testing.assert_close(
        ninetoothed_output,
        reference_output,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )
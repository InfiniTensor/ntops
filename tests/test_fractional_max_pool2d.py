import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


_TEST_CASES_DATA = [
    ((2, 3, 15, 15), None, (3, 3), (5, 5), False),
    ((1, 4, 16, 14), (896, 224, 14, 1), (4, 3), (4, 5), False),
    ((2, 2, 17, 19), None, (5, 5), (7, 6), False),
    ((3, 6, 9, 11), None, (2, 2), (4, 5), False),
    ((1, 8, 20, 20), (3200, 400, 20, 1), (3, 3), (6, 6), False),
    ((2, 5, 12, 10), None, (4, 3), (3, 3), False),
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
def test_fractional_max_pool2d_cases(
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

    n, c, h, w = in_shape

    random_samples = torch.rand(
        (n, c, 2),
        dtype=dtype,
        device=input.device,
    )

    ninetoothed_output = ntops.torch.fractional_max_pool2d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=None,
        return_indices=return_indices,
        _random_samples=random_samples,
    )

    reference_output = F.fractional_max_pool2d(
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
@pytest.mark.parametrize("kernel_size", (2, (3, 3)))
@pytest.mark.parametrize(
    "output_size, output_ratio",
    (
        ((50, 50), None),
        (None, (0.5, 0.5)),
    ),
)
@pytest.mark.parametrize("n, c, h, w", ((2, 3, 112, 112),))
def test_fractional_max_pool2d_ratio_and_size(
    n,
    c,
    h,
    w,
    kernel_size,
    output_size,
    output_ratio,
    dtype,
    device,
):
    input = torch.randn(
        (n, c, h, w),
        dtype=dtype,
        device=device,
    )

    random_samples = torch.rand(
        (n, c, 2),
        dtype=dtype,
        device=input.device,
    )

    ninetoothed_output = ntops.torch.fractional_max_pool2d(
        input,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=False,
        _random_samples=random_samples,
    )

    reference_output = F.fractional_max_pool2d(
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
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=1e-3, atol=1e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape",
    [(0,), (1,), (7,), (3, 5), (2, 3, 4), (1, 4, 1), (8, 1)],
)
def test_rad2deg_float_shapes(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")

    output = ntops.torch.rad2deg(input)
    reference = torch.rad2deg(input)

    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_rad2deg_integer_promotes_to_float32(dtype):
    input = torch.tensor([0, 1, -2, 7], dtype=dtype, device="cuda")

    output = ntops.torch.rad2deg(input)
    reference = torch.rad2deg(input)

    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_non_contiguous_and_out():
    base = torch.randn((5, 7), dtype=torch.float32, device="cuda")
    input = base.t()
    out = torch.empty_like(input)

    result = ntops.torch.rad2deg(input, out=out)
    reference = torch.rad2deg(input)

    assert result is out
    _assert_close(out, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_3d_permute_non_contiguous_and_out():
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    out = torch.empty_like(input)

    result = ntops.torch.rad2deg(input, out=out)
    reference = torch.rad2deg(input)

    assert result is out
    _assert_close(out, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_scalar():
    input = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    output = ntops.torch.rad2deg(input)
    reference = torch.rad2deg(input)

    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_special_values():
    input = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        dtype=torch.float32,
        device="cuda",
    )

    output = ntops.torch.rad2deg(input)
    reference = torch.rad2deg(input)

    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_resizes_out_like_torch():
    input = torch.randn((2, 3), dtype=torch.float32, device="cuda")
    out = torch.empty((1,), dtype=torch.float32, device="cuda")

    with pytest.warns(UserWarning):
        result = ntops.torch.rad2deg(input, out=out)
    reference = torch.rad2deg(input)

    assert result is out
    _assert_close(out, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_rad2deg_rejects_integer_out_for_float_result():
    input = torch.tensor([1], dtype=torch.int32, device="cuda")
    out = torch.empty_like(input)

    with pytest.raises(RuntimeError):
        ntops.torch.rad2deg(input, out=out)

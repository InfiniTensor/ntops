import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal_with_nan(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.equal(torch.isnan(output), torch.isnan(reference))
    mask = ~torch.isnan(reference)
    assert torch.equal(output[mask], reference[mask])


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "input_shape, other_shape",
    [
        ((0,), (0,)),
        ((9,), (9,)),
        ((4, 1), (1, 7)),
        ((2, 3, 4), (1, 3, 1)),
        ((4, 1), (3,)),
        ((2, 1, 4), (1, 3, 1)),
    ],
)
def test_copysign_float_shapes(dtype, input_shape, other_shape):
    input = torch.randn(input_shape, dtype=dtype, device="cuda")
    other = torch.randn(other_shape, dtype=dtype, device="cuda")

    output = ntops.torch.copysign(input, other)
    reference = torch.copysign(input, other)

    _assert_equal_with_nan(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "input_dtype, other_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int16, torch.float32),
        (torch.float16, torch.int16),
        (torch.bool, torch.int32),
    ],
)
def test_copysign_promotes_like_torch(input_dtype, other_dtype):
    input = torch.tensor([-1, 2, -3], dtype=input_dtype, device="cuda")
    other = torch.tensor([1, -1, -2], dtype=other_dtype, device="cuda")

    output = ntops.torch.copysign(input, other)
    reference = torch.copysign(input, other)

    _assert_equal_with_nan(output, reference)


@skip_if_cuda_not_available
def test_copysign_non_contiguous_and_out():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    other = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    out = torch.empty_like(input)

    result = ntops.torch.copysign(input, other, out=out)
    reference = torch.copysign(input, other)

    assert result is out
    _assert_equal_with_nan(out, reference)


@skip_if_cuda_not_available
def test_copysign_3d_permute_non_contiguous_and_out():
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    other = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    out = torch.empty_like(input)

    result = ntops.torch.copysign(input, other, out=out)
    reference = torch.copysign(input, other)

    assert result is out
    _assert_equal_with_nan(out, reference)


@skip_if_cuda_not_available
def test_copysign_scalar():
    input = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    other = torch.tensor(-1.0, dtype=torch.float32, device="cuda")

    output = ntops.torch.copysign(input, other)
    reference = torch.copysign(input, other)

    _assert_equal_with_nan(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype, view_dtype",
    [(torch.float16, torch.int16), (torch.float32, torch.int32), (torch.float64, torch.int64)],
)
def test_copysign_special_sign_bits(dtype, view_dtype):
    input = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=dtype, device="cuda")
    other = torch.tensor([-0.0, 0.0, -1.0, 1.0, -0.0], dtype=dtype, device="cuda")

    output = ntops.torch.copysign(input, other)
    reference = torch.copysign(input, other)

    assert torch.equal(torch.isnan(output), torch.isnan(reference))
    mask = ~torch.isnan(reference)
    assert torch.equal(output[mask].view(view_dtype), reference[mask].view(view_dtype))


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
def test_copysign_resizes_out_like_torch(dtype):
    input = torch.randn((2, 3), dtype=dtype, device="cuda")
    other = torch.randn((2, 3), dtype=dtype, device="cuda")
    out = torch.empty((1,), dtype=dtype, device="cuda")

    with pytest.warns(UserWarning):
        result = ntops.torch.copysign(input, other, out=out)
    reference = torch.copysign(input, other)

    assert result is out
    _assert_equal_with_nan(out, reference)


@skip_if_cuda_not_available
def test_copysign_rejects_integer_out_for_float_result():
    input = torch.tensor([-1, 2], dtype=torch.int32, device="cuda")
    other = torch.tensor([1, -1], dtype=torch.int32, device="cuda")
    out = torch.empty_like(input)

    with pytest.raises(RuntimeError):
        ntops.torch.copysign(input, other, out=out)

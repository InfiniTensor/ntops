import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_nextafter_equal(output, reference):
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
    ],
)
def test_nextafter_float_shapes(dtype, input_shape, other_shape):
    input = torch.randn(input_shape, dtype=dtype, device="cuda")
    other = torch.randn(other_shape, dtype=dtype, device="cuda")

    output = ntops.torch.nextafter(input, other)
    reference = torch.nextafter(input, other)

    _assert_nextafter_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_nextafter_special_value_grid(dtype):
    if dtype == torch.float16:
        values = [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 65504.0, -65504.0]
    else:
        values = [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 1e-37, -1e-37]

    input = torch.tensor(values, dtype=dtype, device="cuda").repeat_interleave(len(values))
    other = torch.tensor(values, dtype=dtype, device="cuda").repeat(len(values))

    output = ntops.torch.nextafter(input, other)
    reference = torch.nextafter(input, other)

    _assert_nextafter_equal(output, reference)


@skip_if_cuda_not_available
def test_nextafter_scalar():
    input = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    other = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    output = ntops.torch.nextafter(input, other)
    reference = torch.nextafter(input, other)

    _assert_nextafter_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "input_dtype, other_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float16, torch.float64),
        (torch.float32, torch.float16),
        (torch.float32, torch.float64),
        (torch.float64, torch.float16),
        (torch.float64, torch.float32),
    ],
)
def test_nextafter_mixed_float_dtype_promotes_like_torch(input_dtype, other_dtype):
    input = torch.tensor([1.0], dtype=input_dtype, device="cuda")
    other = torch.tensor([2.0], dtype=other_dtype, device="cuda")

    output = ntops.torch.nextafter(input, other)
    reference = torch.nextafter(input, other)

    _assert_nextafter_equal(output, reference)


@skip_if_cuda_not_available
def test_nextafter_integer_unsupported():
    input = torch.tensor([1], dtype=torch.int32, device="cuda")
    other = torch.tensor([2], dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError):
        ntops.torch.nextafter(input, other)


@skip_if_cuda_not_available
def test_nextafter_non_contiguous_and_out():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    other = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    out = torch.empty_like(input)

    result = ntops.torch.nextafter(input, other, out=out)
    reference = torch.nextafter(input, other)

    assert result is out
    _assert_nextafter_equal(out, reference)


@skip_if_cuda_not_available
def test_nextafter_3d_permute_non_contiguous_and_out():
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    other = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    out = torch.empty_like(input)

    result = ntops.torch.nextafter(input, other, out=out)
    reference = torch.nextafter(input, other)

    assert result is out
    _assert_nextafter_equal(out, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nextafter_resizes_out_like_torch(dtype):
    input = torch.randn((2, 3), dtype=dtype, device="cuda")
    other = torch.randn((2, 3), dtype=dtype, device="cuda")
    out = torch.empty((1,), dtype=dtype, device="cuda")

    with pytest.warns(UserWarning):
        result = ntops.torch.nextafter(input, other, out=out)
    reference = torch.nextafter(input, other)

    assert result is out
    _assert_nextafter_equal(out, reference)


@skip_if_cuda_not_available
def test_nextafter_rejects_integer_out_for_float_result():
    input = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([2.0], dtype=torch.float32, device="cuda")
    out = torch.empty((1,), dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError):
        ntops.torch.nextafter(input, other, out=out)

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "input_shape, other_shape",
    [((0,), (0,)), ((11,), (11,)), ((4, 1), (1, 7)), ((2, 3, 4), (1, 3, 1))],
)
def test_lcm_integer_shapes(dtype, input_shape, other_shape):
    low = 0 if dtype == torch.uint8 else -20
    input = torch.randint(low, 21, input_shape, dtype=dtype, device="cuda")
    other = torch.randint(low, 21, other_shape, dtype=dtype, device="cuda")

    output = ntops.torch.lcm(input, other)
    reference = torch.lcm(input, other)

    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
def test_lcm_zero_and_sign_cases():
    input = torch.tensor([0, 6, -4, -9, 21], dtype=torch.int32, device="cuda")
    other = torch.tensor([3, 4, -6, 0, -6], dtype=torch.int32, device="cuda")

    output = ntops.torch.lcm(input, other)
    reference = torch.lcm(input, other)

    assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype, lhs, rhs",
    [
        (torch.int8, -128, -1),
        (torch.int16, -32768, -1),
        (torch.int32, -2147483648, -1),
        (torch.int64, -9223372036854775808, -1),
    ],
)
def test_lcm_min_integer_overflow_cases(dtype, lhs, rhs):
    input = torch.tensor([lhs], dtype=dtype, device="cuda")
    other = torch.tensor([rhs], dtype=dtype, device="cuda")

    output = ntops.torch.lcm(input, other)
    reference = torch.lcm(input, other)

    assert torch.equal(output, reference)


@skip_if_cuda_not_available
def test_lcm_worst_case_euclid_inputs():
    cases = [
        (torch.int16, 28657, 17711),
        (torch.int32, 1836311903, 1134903170),
        (torch.int64, 7540113804746346429, 4660046610375530309),
    ]

    for dtype, lhs, rhs in cases:
        input = torch.tensor([lhs], dtype=dtype, device="cuda")
        other = torch.tensor([rhs], dtype=dtype, device="cuda")

        output = ntops.torch.lcm(input, other)
        reference = torch.lcm(input, other)

        assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "input_dtype, other_dtype",
    [
        (torch.int16, torch.int32),
        (torch.int32, torch.int64),
        (torch.uint8, torch.int16),
    ],
)
def test_lcm_mixed_dtype_promotes_like_torch(input_dtype, other_dtype):
    input = torch.tensor([6, 1], dtype=input_dtype, device="cuda")
    other = torch.tensor([4, 6], dtype=other_dtype, device="cuda")

    output = ntops.torch.lcm(input, other)
    reference = torch.lcm(input, other)

    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
def test_lcm_bool_bool_unsupported():
    input = torch.tensor([True, False], device="cuda")
    other = torch.tensor([True, True], device="cuda")

    with pytest.raises(NotImplementedError):
        ntops.torch.lcm(input, other)


@skip_if_cuda_not_available
def test_lcm_float_unsupported():
    input = torch.tensor([1.0, 2.0], device="cuda")
    other = torch.tensor([1.0, 3.0], device="cuda")

    with pytest.raises(NotImplementedError):
        ntops.torch.lcm(input, other)


@skip_if_cuda_not_available
def test_lcm_non_contiguous_and_out():
    input = torch.randint(-20, 21, (5, 7), dtype=torch.int32, device="cuda").t()
    other = torch.randint(-20, 21, (5, 7), dtype=torch.int32, device="cuda").t()
    out = torch.empty_like(input)

    result = ntops.torch.lcm(input, other, out=out)
    reference = torch.lcm(input, other)

    assert result is out
    assert torch.equal(out, reference)


@skip_if_cuda_not_available
def test_lcm_3d_permute_non_contiguous_and_out():
    input = torch.randint(-20, 21, (3, 5, 7), dtype=torch.int32, device="cuda").permute(2, 0, 1)
    other = torch.randint(-20, 21, (3, 5, 7), dtype=torch.int32, device="cuda").permute(2, 0, 1)
    out = torch.empty_like(input)

    result = ntops.torch.lcm(input, other, out=out)
    reference = torch.lcm(input, other)

    assert result is out
    assert torch.equal(out, reference)


@skip_if_cuda_not_available
def test_lcm_resizes_out_like_torch():
    input = torch.randint(1, 10, (2, 3), dtype=torch.int32, device="cuda")
    other = torch.randint(1, 10, (2, 3), dtype=torch.int32, device="cuda")
    out = torch.empty((1,), dtype=torch.int32, device="cuda")

    with pytest.warns(UserWarning):
        result = ntops.torch.lcm(input, other, out=out)
    reference = torch.lcm(input, other)

    assert result is out
    assert torch.equal(out, reference)

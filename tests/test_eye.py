"""
eye 算子测试脚本
"""
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_eye_3x3():
    """Test 3x3 identity matrix"""
    result = ntops.torch.eye(3, dtype=torch.float32, device="cuda")
    expected = torch.eye(3, dtype=torch.float32, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_2x4():
    """Test 2x4 rectangular matrix"""
    result = ntops.torch.eye(2, 4, dtype=torch.float32, device="cuda")
    expected = torch.eye(2, 4, dtype=torch.float32, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_5x3():
    """Test 5x3 rectangular matrix (more rows than columns)"""
    result = ntops.torch.eye(5, 3, dtype=torch.float32, device="cuda")
    expected = torch.eye(5, 3, dtype=torch.float32, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_1x1():
    """Test 1x1 matrix"""
    result = ntops.torch.eye(1, dtype=torch.float32, device="cuda")
    expected = torch.eye(1, dtype=torch.float32, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_float16():
    """Test with float16 dtype"""
    result = ntops.torch.eye(3, dtype=torch.float16, device="cuda")
    expected = torch.eye(3, dtype=torch.float16, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_float64():
    """Test with float64 dtype"""
    result = ntops.torch.eye(3, dtype=torch.float64, device="cuda")
    expected = torch.eye(3, dtype=torch.float64, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_invalid_negative():
    """Test that negative dimensions raise ValueError"""
    try:
        ntops.torch.eye(-1, device="cuda")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-negative" in str(e)


@skip_if_cuda_not_available
def test_eye_default_dtype():
    """Test that default dtype is float32"""
    result = ntops.torch.eye(2, device="cuda")
    assert result.dtype == torch.float32


@skip_if_cuda_not_available
def test_eye_large():
    """Test large identity matrix"""
    n = 100
    result = ntops.torch.eye(n, dtype=torch.float32, device="cuda")
    expected = torch.eye(n, dtype=torch.float32, device="cuda")

    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_eye_diagonal_correctness():
    """Verify that only diagonal elements are 1"""
    result = ntops.torch.eye(5, 5, dtype=torch.float32, device="cuda")

    # Check diagonal
    for i in range(5):
        assert result[i, i].item() == pytest.approx(1.0, abs=1e-5)

    # Check off-diagonal
    for i in range(5):
        for j in range(5):
            if i != j:
                assert result[i, j].item() == pytest.approx(0.0, abs=1e-5)

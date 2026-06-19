"""
repeat 算子测试脚本
"""
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_repeat_basic():
    """Test basic repeat functionality"""
    x = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.float32)
    result = ntops.torch.repeat(x, (2, 3))
    expected = x.repeat(2, 3)

    assert result.shape == expected.shape == (4, 6)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_1d():
    """Test repeating 1D tensor"""
    x = torch.tensor([1, 2, 3], device="cuda", dtype=torch.float32)
    result = ntops.torch.repeat(x, (4,))

    assert result.shape == (12,)
    assert torch.equal(result, torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], device="cuda", dtype=torch.float32))


@skip_if_cuda_not_available
def test_repeat_3d():
    """Test repeating 3D tensor"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.repeat(x, (2, 1, 3))

    assert result.shape == (4, 3, 12)
    expected = x.repeat(2, 1, 3)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_single_dim():
    """Test repeating along single dimension"""
    x = torch.randn(3, 5, device="cuda")
    result = ntops.torch.repeat(x, (1, 4))

    assert result.shape == (3, 20)
    expected = x.repeat(1, 4)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_no_repeat():
    """Test with repeats of 1 (no actual repetition)"""
    x = torch.randn(2, 3, device="cuda")
    result = ntops.torch.repeat(x, (1, 1))

    assert result.shape == (2, 3)
    expected = x.repeat(1, 1)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_large():
    """Test with large repeat factors"""
    x = torch.randn(2, 2, device="cuda")
    result = ntops.torch.repeat(x, (10, 10))

    assert result.shape == (20, 20)
    expected = x.repeat(10, 10)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_dtype_preservation():
    """Test that dtype is preserved"""
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(2, 3, device="cuda", dtype=dtype)
        result = ntops.torch.repeat(x, (2, 1))
        assert result.dtype == dtype


@skip_if_cuda_not_available
def test_repeat_device_preservation():
    """Test that device is preserved"""
    x = torch.randn(2, 3, device="cuda")
    result = ntops.torch.repeat(x, (2, 3))
    assert result.device.type == "cuda"


@skip_if_cuda_not_available
def test_repeat_gradient():
    """Test that gradients flow through repeat correctly"""
    x = torch.randn(2, 3, device="cuda", requires_grad=True)
    result = ntops.torch.repeat(x, (2, 3))

    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Each element contributes to 6 positions (2 * 3), so gradient is 6
    assert torch.allclose(x.grad, torch.full_like(x, 6.0))


@skip_if_cuda_not_available
def test_repeat_invalid_repeats_length():
    """Test that invalid repeats length raises ValueError"""
    x = torch.randn(2, 3, device="cuda")

    with pytest.raises(ValueError, match="repeats length.*must match"):
        ntops.torch.repeat(x, (2, 3, 4))  # 3 repeats for 2D tensor


@skip_if_cuda_not_available
def test_repeat_single_element():
    """Test repeating single element tensor"""
    x = torch.tensor([5.0], device="cuda")
    result = ntops.torch.repeat(x, (10,))

    assert result.shape == (10,)
    assert torch.all(result == 5.0)


@skip_if_cuda_not_available
def test_repeat_4d_tensor():
    """Test repeating 4D tensor"""
    x = torch.randn(2, 3, 4, 5, device="cuda")
    result = ntops.torch.repeat(x, (1, 2, 1, 3))

    assert result.shape == (2, 6, 4, 15)
    expected = x.repeat(1, 2, 1, 3)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_repeat_data_correctness():
    """Verify that repeated data is correct"""
    x = torch.arange(6, device="cuda").reshape(2, 3)
    result = ntops.torch.repeat(x, (2, 2))

    # Shape should be (4, 6)
    assert result.shape == (4, 6)

    # Check some specific values
    # Original:
    # [[0, 1, 2],
    #  [3, 4, 5]]
    # After repeat(2, 2):
    # [[0, 1, 2, 0, 1, 2],
    #  [3, 4, 5, 3, 4, 5],
    #  [0, 1, 2, 0, 1, 2],
    #  [3, 4, 5, 3, 4, 5]]

    assert result[0, 0].item() == 0
    assert result[0, 3].item() == 0
    assert result[1, 0].item() == 3
    assert result[3, 5].item() == 5


@skip_if_cuda_not_available
def test_repeat_tuple_input():
    """Test that tuple input works correctly"""
    x = torch.randn(2, 3, device="cuda")
    result_tuple = ntops.torch.repeat(x, (2, 3))
    result_list = ntops.torch.repeat(x, [2, 3])

    assert torch.equal(result_tuple, result_list)


@skip_if_cuda_not_available
def test_repeat_with_zeros():
    """Test repeat with 0 in some dimensions (edge case)"""
    x = torch.randn(2, 3, device="cuda")
    # PyTorch repeat with 0 results in empty tensor
    result = ntops.torch.repeat(x, (0, 1))

    assert result.shape == (0, 3)
    assert result.numel() == 0

"""
rad2deg 算子精度验证测试

按照 ninetoothed-skill 测试生成协议编写
"""
import math
import pytest
import torch
import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_rad2deg_basic(dtype, rtol, atol):
    """基本功能测试 - 常见弧度值"""
    device = torch.device("cuda")

    # 测试常见弧度值
    test_radians = torch.tensor([
        0.0,                    # 0 度
        math.pi / 6,            # 30 度
        math.pi / 4,            # 45 度
        math.pi / 3,            # 60 度
        math.pi / 2,            # 90 度
        math.pi,                # 180 度
        2 * math.pi,            # 360 度
        -math.pi / 4,           # -45 度
    ], dtype=dtype, device=device)

    # ntops 结果
    ntops_result = ntops.torch.rad2deg(test_radians)

    # 参考结果
    reference = test_radians * (180.0 / math.pi)

    # 四项必检
    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol), \
        f"精度不匹配: max_diff={(ntops_result - reference).abs().max().item()}"
    assert not torch.isnan(ntops_result).any(), "存在 NaN"
    assert not torch.isinf(ntops_result).any(), "存在 Inf"


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_rad2deg_medium(dtype, rtol, atol):
    """中等规模测试"""
    device = torch.device("cuda")

    input_tensor = torch.randn(64, 64, dtype=dtype, device=device)

    ntops_result = ntops.torch.rad2deg(input_tensor)
    reference = input_tensor * (180.0 / math.pi)

    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(ntops_result).any()
    assert not torch.isinf(ntops_result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_rad2deg_large(dtype, rtol, atol):
    """大规模测试"""
    device = torch.device("cuda")

    input_tensor = torch.randn(1024, 1024, dtype=dtype, device=device)

    ntops_result = ntops.torch.rad2deg(input_tensor)
    reference = input_tensor * (180.0 / math.pi)

    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(ntops_result).any()
    assert not torch.isinf(ntops_result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_rad2deg_edge_cases(dtype, rtol, atol):
    """边界情况测试"""
    device = torch.device("cuda")

    # 测试 1D 张量（17 不整除常见 block_size）
    tensor_1d = torch.randn(17, dtype=dtype, device=device)
    ntops_result = ntops.torch.rad2deg(tensor_1d)
    reference = tensor_1d * (180.0 / math.pi)
    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)

    # 测试 3D 张量
    tensor_3d = torch.randn(8, 16, 32, dtype=dtype, device=device)
    ntops_result = ntops.torch.rad2deg(tensor_3d)
    reference = tensor_3d * (180.0 / math.pi)
    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)

    # 测试 5D 张量
    tensor_5d = torch.randn(2, 4, 8, 16, 32, dtype=dtype, device=device)
    ntops_result = ntops.torch.rad2deg(tensor_5d)
    reference = tensor_5d * (180.0 / math.pi)
    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_rad2deg_non_contiguous(dtype, rtol, atol):
    """非连续输入测试（转置、切片）"""
    device = torch.device("cuda")

    # 测试转置张量
    tensor = torch.randn(32, 64, dtype=dtype, device=device)
    transposed = tensor.t()  # 转置后非连续

    ntops_result = ntops.torch.rad2deg(transposed)
    reference = transposed * (180.0 / math.pi)

    assert torch.allclose(ntops_result, reference, rtol=rtol, atol=atol)


def test_rad2deg_special_values():
    """特殊值测试"""
    device = torch.device("cuda")

    # 测试零值
    zero = torch.zeros(10, dtype=torch.float32, device=device)
    ntops_result = ntops.torch.rad2deg(zero)
    assert torch.allclose(ntops_result, zero, rtol=1e-5, atol=1e-5)

    # 测试负值
    negative = -torch.ones(10, dtype=torch.float32, device=device) * math.pi
    ntops_result = ntops.torch.rad2deg(negative)
    reference = negative * (180.0 / math.pi)
    assert torch.allclose(ntops_result, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

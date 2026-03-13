import pytest
import torch
import random

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

m = torch.nn.Hardsigmoid()


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("inplace", [False, True], ids=["not_inplace", "inplace"])
def test_hardsigmoid(shape, dtype, device, rtol, atol, inplace):
    input = torch.randn(shape, dtype=dtype, device=device)
    input_copy = input.clone()
    input_id = id(input)

    output = ntops.torch.hardsigmoid(input, inplace=inplace)

    if inplace:
        # 验证 inplace 返回同一个对象
        assert id(output) == input_id, "inplace=True should return the same tensor"
    else:
        # 验证非 inplace 不修改输入
        torch.testing.assert_close(input, input_copy, rtol=0, atol=0)

    # 验证结果正确
    reference = m(input_copy)
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)

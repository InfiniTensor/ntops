import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import _random_shape


def generate_argmin_arguments():
    """专门为 argmin 生成参数"""
    arguments = []
    dtype_arr = (torch.float32, torch.float16)

    for ndim in range(1, 5):
        for dtype in dtype_arr:
            device = "cuda"
            atol = 0.001 if dtype is torch.float32 else 0.01
            rtol = 0.001 if dtype is torch.float32 else 0.01

            shape = _random_shape(ndim)

            # axis=None 的情况
            for keepdim in [False, True]:
                arguments.append((shape, None, keepdim, dtype, device, rtol, atol))

            # axis=0 到 ndim-1 的情况
            for axis in range(ndim):
                for keepdim in [False, True]:
                    arguments.append((shape, axis, keepdim, dtype, device, rtol, atol))

    return "shape, axis, keepdim, dtype, device, rtol, atol", arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_argmin_arguments())
def test_argmin(shape, axis, keepdim, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    output = ntops.torch.argmin(input, axis=axis, keepdim=keepdim)
    reference = torch.argmin(input, dim=axis, keepdim=keepdim)
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)

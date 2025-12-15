import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


# Test cases: (in_shape, output_size, kernel_size, dilation, stride, padding)
_TEST_CASES = [
    ((2, 27, 36), (8, 8), (3, 3), (1, 1), (1, 1), (0, 0)),
    ((2, 32, 16), (16, 16), (4, 4), (1, 1), (4, 4), (0, 0)),
    ((3, 36, 40), (7, 9), (3, 2), (1, 1), (1, 1), (0, 0)),
    ((2, 45, 20), (12, 6), (3, 3), (1, 1), (2, 1), (0, 0)),

    # padding 在 infinicore 层面处理
    # 原来对应 ((1,4,10,12), None, (5,3), 1, 1, (2,1))
    # L = 4 * 12 = 48, channels = 4*5*3 = 60
    # ((1, 60, 48), (10, 12), (5, 3), (1, 1), (2, 1), (1, 1)),
    # 原来对应 ((1,8,9,11), None, (2,3), 1, 1, (1,2))
    # L = 10 * 6 = 60, channels = 8*2*3 = 48
    # ((1, 48, 60), (9, 11), (2, 3), (1, 1), (1, 2), (1, 1)),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "in_shape, output_size, kernel_size, dilation, stride, padding",
    _TEST_CASES,
)
def test_fold(in_shape, output_size, kernel_size, dilation, stride, padding, dtype):
    device = "cuda"

    x = torch.randn(*in_shape, dtype=dtype, device=device)

    reference_output = torch.nn.functional.fold(
        x,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    ninetoothed_output = ntops.torch.fold(
        x,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    if dtype is torch.float32:
        atol = 0.001
        rtol = 0.001
    else:
        atol = 0.01
        rtol = 0.01

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

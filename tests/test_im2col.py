import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
    ),
)
@pytest.mark.parametrize("dilation", (1, 2, (2, 3)))
@pytest.mark.parametrize("padding", (0, 1, (2, 3)))
@pytest.mark.parametrize("stride", (1, 2, (2, 3)))
@pytest.mark.parametrize("kernel_size", ((1, 1), (3, 3)))
@pytest.mark.parametrize("n, c, h, w", ((2, 3, 112, 112),))
def test_im2col(
    n,
    c,
    h,
    w,
    kernel_size,
    stride,
    padding,
    dilation,
    dtype,
    device,
    rtol,
    atol,
):
    input = torch.randn((n, c, h, w), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.im2col(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    reference_output = F.unfold(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
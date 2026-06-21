import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


_KL_DIV_TEST_CASES = [
    ((4, 5), "batchmean", False),
    ((8, 8), "sum", False),
    ((1, 10), "batchmean", True),
    ((16, 100), "batchmean", False),
    ((3, 7), "batchmean", False),
    ((2, 2), "sum", False),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("case_shape,reduction,log_target", _KL_DIV_TEST_CASES)
def test_kl_div(shape, dtype, device, rtol, atol, case_shape, reduction, log_target):
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        pytest.skip("kl_div test only covers float32, float16, and bfloat16")

    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-5
    elif dtype == torch.float16:
        rtol, atol = 1e-1, 1e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 5e-2, 1e-2

    if log_target:
        input = torch.randn(case_shape, dtype=dtype, device=device)
        target = torch.randn(case_shape, dtype=dtype, device=device)
    else:
        input = torch.randn(case_shape, dtype=dtype, device=device)
        target = torch.rand(case_shape, dtype=dtype, device=device) + 0.1

    ninetoothed_output = ntops.torch.kl_div(
        input,
        target,
        reduction=reduction,
        log_target=log_target,
    )

    reference_output = F.kl_div(
        input,
        target,
        reduction=reduction,
        log_target=log_target,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )
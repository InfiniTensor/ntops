import random

import pytest
import torch

import ntops
from tests.utils import generate_arguments


_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def _tolerance(dtype, rtol, atol):
    if dtype == torch.float16:
        return max(rtol, 1e-2), max(atol, 1e-3)
    if dtype == torch.bfloat16:
        return max(rtol, 5e-2), max(atol, 1e-2)
    return max(rtol, 1e-4), max(atol, 1e-5)


@pytest.mark.parametrize(*generate_arguments())
def test_gumbel_softmax(shape, dtype, device, rtol, atol):
    if dtype not in _FLOAT_DTYPES:
        return

    if len(shape) == 0:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    dim = random.randint(0, input.ndim - 1)
    tau = random.choice([0.5, 1.0, 1.5])
    hard = random.choice([False, True])

    output = ntops.torch.gumbel_softmax(
        input,
        tau=tau,
        hard=hard,
        dim=dim,
    )

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert output.device == input.device

    output_fp32 = output.to(torch.float32)

    assert torch.isfinite(output_fp32).all()

    rtol, atol = _tolerance(dtype, rtol, atol)

    sum_output = output_fp32.sum(dim=dim)
    expected_sum = torch.ones_like(sum_output)

    assert torch.allclose(sum_output, expected_sum, rtol=rtol, atol=atol)

    if hard:
        assert ((output_fp32 == 0.0) | (output_fp32 == 1.0)).all()
    else:
        assert (output_fp32 >= 0.0).all()
        assert (output_fp32 <= 1.0).all()
import math
import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    p = random.uniform(0.05, 0.9)

    # PyTorch feature_alpha_dropout 不支持 1D 输入
    if input.ndim < 2:
        with pytest.raises(RuntimeError):
            F.feature_alpha_dropout(
                input,
                p=p,
                training=True,
            )

        with pytest.raises(AssertionError):
            ntops.torch.feature_alpha_dropout(
                input,
                p=p,
                training=True,
            )

        return

    ninetoothed_output = ntops.torch.feature_alpha_dropout(
        input,
        p=p,
        training=True,
    )

    reference_output = F.feature_alpha_dropout(
        input,
        p=p,
        training=True,
    )

    assert ninetoothed_output.shape == reference_output.shape

    alpha_prime = -1.7580993408473766
    q = 1.0 - p
    a = 1.0 / math.sqrt(q * (1.0 + p * alpha_prime * alpha_prime))
    b = -a * p * alpha_prime
    drop_value = alpha_prime * a + b

    drop_value = torch.tensor(
        drop_value,
        dtype=ninetoothed_output.dtype,
        device=ninetoothed_output.device,
    )

    ninetoothed_dropped = torch.isclose(
        ninetoothed_output,
        drop_value,
        rtol=rtol,
        atol=max(atol, 1e-3),
    )

    reference_dropped = torch.isclose(
        reference_output,
        drop_value,
        rtol=rtol,
        atol=max(atol, 1e-3),
    )

    ninetoothed_drop_ratio = ninetoothed_dropped.sum().item() / input.numel()
    reference_drop_ratio = reference_dropped.sum().item() / input.numel()

    assert abs(ninetoothed_drop_ratio - reference_drop_ratio) < 0.1

    kept = ~ninetoothed_dropped

    assert torch.allclose(
        ninetoothed_output[kept],
        input[kept] * a + b,
        rtol=rtol,
        atol=max(atol, 1e-3),
    )
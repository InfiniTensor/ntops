import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_embedding(shape, dtype, device, rtol, atol):
    vocab_size = 10000
    embedding_dim = 2048

    input = torch.randint(0, vocab_size, shape, device=device)
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype, device=device)

    # among optional params, only max_norm and norm_type are supported, because only they affect output, others are for gradient calculation
    ninetoothed_output = ntops.torch.embedding(
        input, weight, max_norm=None, norm_type=1.1
    )
    reference_output = torch.nn.functional.embedding(
        input,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=1.1,
        scale_grad_by_freq=False,
        sparse=False,
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

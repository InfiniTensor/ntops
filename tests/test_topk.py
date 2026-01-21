import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import _random_shape


def generate_topk_args_dim():
    args = []
    for dtype in (torch.float32, torch.float16):
        device = "cuda"
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (1e-2, 1e-2)

        for ndim in range(1, 4):
            for _ in range(5):
                shape = _random_shape(ndim)
                dim = random.randint(0, ndim - 1)
                dim_size = shape[dim]

                k = random.randint(1, min(dim_size, 128))
                args.append((shape, k, dim, dtype, device, rtol, atol))
    return "shape, k, dim, dtype, device, rtol, atol", args


def generate_topk_args_global():
    args = []
    for dtype in (torch.float32, torch.float16):
        device = "cuda"
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (1e-2, 1e-2)

        candidates = [(100,), (10, 20), (5, 5, 5)]
        for shape in candidates:
            numel = 1
            for s in shape:
                numel *= s
            k = random.randint(1, min(numel, 64))
            args.append((shape, k, dtype, device, rtol, atol))
    return "shape, k, dtype, device, rtol, atol", args


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_topk_args_dim())
@pytest.mark.parametrize("largest", [True, False])
def test_topk_dim(shape, k, dim, dtype, device, rtol, atol, largest):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_v, ntops_i = ntops.torch.topk(input_tensor, k, dim=dim, largest=largest)
    ref_v, ref_i = torch.topk(input_tensor, k, dim=dim, largest=largest)

    assert torch.allclose(ntops_v, ref_v, rtol=rtol, atol=atol)
    gathered = torch.gather(input_tensor, dim, ntops_i)
    assert torch.allclose(gathered, ntops_v, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_topk_args_global())
@pytest.mark.parametrize("largest", [True, False])
def test_topk_global(shape, k, dtype, device, rtol, atol, largest):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_v, ntops_i = ntops.torch.topk(input_tensor, k, dim=None, largest=largest)

    ref_v, ref_i = torch.topk(input_tensor.flatten(), k, dim=0, largest=largest)

    assert torch.allclose(ntops_v, ref_v, rtol=rtol, atol=atol)

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize(
    "input_shape, index_shape, src_shape",
    [
        ((3, 5), (3, 2), (3, 2)),
        ((4, 4), (4, 3), (4, 3)),
    ],
)
def test_scatter_add(input_shape, index_shape, src_shape, dim, dtype):
    input = torch.randn(input_shape, dtype=dtype, device="cuda")
    src = torch.randn(src_shape, dtype=dtype, device="cuda")
    index = torch.randint(0, input_shape[dim], index_shape, device="cuda")

    nt_out = ntops.torch.scatter_add(input, dim, index, src)
    ref_out = input.clone().scatter_add_(dim, index, src)
    assert torch.allclose(nt_out, ref_out)


@skip_if_cuda_not_available
def test_scatter_add_large():
    """>128 elements — exercises multiple kernel blocks."""
    torch.manual_seed(42)
    input = torch.randn(50, 10, device="cuda")  # 500 elements > 128
    src = torch.randn(50, 10, device="cuda")
    index = torch.randint(0, 50, (50, 10), device="cuda")
    nt_out = ntops.torch.scatter_add(input, 0, index, src)
    ref_out = input.clone().scatter_add_(0, index, src)
    assert torch.allclose(nt_out, ref_out)


@skip_if_cuda_not_available
def test_scatter_add_repeated_dst():
    """Multiple src elements mapping to the same output position."""
    input = torch.zeros(5, device="cuda")
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    index = torch.tensor([2, 2, 2, 2, 2], device="cuda").unsqueeze(0)
    src_2d = src.unsqueeze(0)
    nt_out = ntops.torch.scatter_add(input.unsqueeze(0), 1, index, src_2d)
    ref_out = input.unsqueeze(0).clone().scatter_add_(1, index, src_2d)
    assert torch.allclose(nt_out, ref_out)
    assert nt_out[0, 2].item() == 15.0  # 1+2+3+4+5


@skip_if_cuda_not_available
def test_scatter_add_repeated_calls():
    """Multiple calls with same kernel — no autotune pollution."""
    input = torch.randn(3, 5, device="cuda")
    index = torch.randint(0, 3, (3, 2), device="cuda")
    src = torch.randn(3, 2, device="cuda")
    # Run 3 times, verify consistency
    results = []
    for _ in range(3):
        results.append(ntops.torch.scatter_add(input, 0, index, src))
    for r in results[1:]:
        assert torch.equal(results[0], r)


@skip_if_cuda_not_available
def test_scatter_add_negative_dim():
    """Negative dim normalization."""
    input = torch.randn(3, 5, device="cuda")
    index = torch.randint(0, 5, (3, 2), device="cuda")
    src = torch.randn(3, 2, device="cuda")
    nt_out = ntops.torch.scatter_add(input, -1, index, src)
    ref_out = input.clone().scatter_add_(-1, index, src)
    assert torch.allclose(nt_out, ref_out)

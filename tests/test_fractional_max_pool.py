import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kernel_size", [(2, 2), (3, 3)])
@pytest.mark.parametrize("n,c,h,w,output_size", [(1, 2, 8, 8, (4, 4)), (1, 2, 10, 10, (5, 5))])
def test_fractional_max_pool2d(n, c, h, w, kernel_size, output_size, dtype):
    torch.manual_seed(42)
    input = torch.randn(n, c, h, w, dtype=dtype, device="cuda")

    # Use fixed random samples so both calls produce identical results
    _random_samples = torch.rand(n, c, 2, device="cuda")

    nt_out = ntops.torch.fractional_max_pool2d(
        input, kernel_size, output_size=output_size, _random_samples=_random_samples
    )
    ref_out = torch.nn.functional.fractional_max_pool2d(
        input, kernel_size, output_size=output_size, _random_samples=_random_samples
    )
    assert torch.allclose(nt_out, ref_out)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kernel_size", [(2, 2, 2)])
@pytest.mark.parametrize("n,c,d,h,w,output_size", [(1, 2, 8, 8, 8, (4, 4, 4))])
def test_fractional_max_pool3d(n, c, d, h, w, kernel_size, output_size, dtype):
    torch.manual_seed(42)
    input = torch.randn(n, c, d, h, w, dtype=dtype, device="cuda")
    _random_samples = torch.rand(n, c, 3, device="cuda")

    nt_out = ntops.torch.fractional_max_pool3d(
        input, kernel_size, output_size=output_size, _random_samples=_random_samples
    )
    ref_out = torch.nn.functional.fractional_max_pool3d(
        input, kernel_size, output_size=output_size, _random_samples=_random_samples
    )
    assert torch.allclose(nt_out, ref_out)


# ---- Edge case tests ----


@skip_if_cuda_not_available
def test_fractional_max_pool2d_output_size_1():
    """output_size=1: last-window branch, alpha division skipped."""
    torch.manual_seed(42)
    inp = torch.randn(1, 2, 6, 6, device="cuda")
    rs = torch.rand(1, 2, 2, device="cuda")
    nt = ntops.torch.fractional_max_pool2d(inp, (2, 2), output_size=(1, 1), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool2d(inp, (2, 2), output_size=(1, 1), _random_samples=rs)
    assert torch.allclose(nt, ref)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_nan():
    """NaN in input should propagate via isnan(val) check."""
    inp = torch.zeros(1, 1, 4, 4, device="cuda")
    inp[0, 0, 1, 1] = float("nan")
    rs = torch.zeros(1, 1, 2, device="cuda")  # sample 0 → all windows start at 0
    nt = ntops.torch.fractional_max_pool2d(inp, (2, 2), output_size=(2, 2), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool2d(inp, (2, 2), output_size=(2, 2), _random_samples=rs)
    # NaN equality needs special handling
    assert torch.allclose(nt[~nt.isnan()], ref[~ref.isnan()], equal_nan=True)
    assert nt.isnan().equal(ref.isnan())


@skip_if_cuda_not_available
@pytest.mark.parametrize("return_indices", [False, True])
def test_fractional_max_pool2d_return_indices(return_indices):
    """return_indices: verify indices are flat spatial offsets."""
    torch.manual_seed(42)
    inp = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4).cuda()
    rs = torch.zeros(1, 1, 2, device="cuda")
    result = ntops.torch.fractional_max_pool2d(
        inp, (2, 2), output_size=(2, 2), _random_samples=rs, return_indices=return_indices
    )
    ref = torch.nn.functional.fractional_max_pool2d(
        inp, (2, 2), output_size=(2, 2), _random_samples=rs, return_indices=return_indices
    )
    if return_indices:
        nt_vals, nt_idx = result
        ref_vals, ref_idx = ref
        assert torch.equal(nt_idx, ref_idx)
        assert torch.equal(nt_vals, ref_vals)
    else:
        assert torch.equal(result, ref)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_unbatched():
    """Unbatched input (C, H, W).  Random samples auto-unsqueezed by wrapper."""
    torch.manual_seed(42)
    inp = torch.randn(2, 8, 8, device="cuda")
    rs = torch.rand(1, 2, 2, device="cuda")  # PyTorch expects (N, C, 2) even for unbatched
    nt = ntops.torch.fractional_max_pool2d(inp, (2, 2), output_size=(4, 4), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool2d(inp, (2, 2), output_size=(4, 4), _random_samples=rs)
    assert torch.allclose(nt, ref)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_larger():
    """Larger batch/channel to exercise tile tail."""
    torch.manual_seed(42)
    inp = torch.randn(3, 5, 6, 6, device="cuda")
    rs = torch.rand(3, 5, 2, device="cuda")
    nt = ntops.torch.fractional_max_pool2d(inp, (2, 2), output_size=(3, 5), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool2d(inp, (2, 2), output_size=(3, 5), _random_samples=rs)
    assert torch.allclose(nt, ref, rtol=1e-3)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_output_ratio():
    """output_ratio instead of output_size."""
    torch.manual_seed(42)
    inp = torch.randn(1, 2, 8, 8, device="cuda")
    rs = torch.rand(1, 2, 2, device="cuda")
    nt = ntops.torch.fractional_max_pool2d(inp, (2, 2), output_ratio=(0.5, 0.5), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool2d(inp, (2, 2), output_ratio=(0.5, 0.5), _random_samples=rs)
    assert torch.allclose(nt, ref)


@skip_if_cuda_not_available
def test_fractional_max_pool3d_larger():
    """3D with different output sizes to exercise tile tail."""
    torch.manual_seed(42)
    inp = torch.randn(2, 3, 6, 6, 6, device="cuda")
    rs = torch.rand(2, 3, 3, device="cuda")
    nt = ntops.torch.fractional_max_pool3d(inp, (2, 2, 2), output_size=(3, 3, 3), _random_samples=rs)
    ref = torch.nn.functional.fractional_max_pool3d(inp, (2, 2, 2), output_size=(3, 3, 3), _random_samples=rs)
    assert torch.allclose(nt, ref, rtol=1e-3)

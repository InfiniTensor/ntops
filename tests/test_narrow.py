import pytest, torch, ntops

def test_narrow_basic():
    x = torch.arange(12, device="cuda").reshape(3, 4)
    for dim, start, length in [(0, 0, 2), (1, 1, 2), (0, 1, 1), (1, 0, 4)]:
        assert torch.equal(ntops.torch.narrow(x, dim, start, length),
                          torch.narrow(x, dim, start, length))

def test_narrow_1d():
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    assert torch.equal(ntops.torch.narrow(x, 0, 2, 2), torch.tensor([3, 4], device="cuda"))

def test_narrow_float16():
    x = torch.randn(10, device="cuda", dtype=torch.float16)
    assert torch.equal(ntops.torch.narrow(x, 0, 3, 4), torch.narrow(x, 0, 3, 4))

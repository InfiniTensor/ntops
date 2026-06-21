import math
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _gcd_cpu(a, b):
    """Reference GCD implementation using math.gcd"""
    a_abs = abs(a)
    b_abs = abs(b)
    if a_abs == 0 and b_abs == 0:
        return 0
    return math.gcd(a_abs, b_abs)


def _lcm_cpu(a, b):
    """Reference LCM implementation"""
    if a == 0 or b == 0:
        return 0
    a_abs = abs(a)
    b_abs = abs(b)
    gcd_val = _gcd_cpu(a, b)
    # Divide first to avoid overflow: (a / gcd) * b
    return (a_abs // gcd_val) * b_abs


@skip_if_cuda_not_available
def test_lcm_int32():
    a = torch.tensor([4, 6, 0, 21, -4, -6], dtype=torch.int32).cuda()
    b = torch.tensor([6, 8, 5, 6, 6, -8], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([_lcm_cpu(4, 6), _lcm_cpu(6, 8), _lcm_cpu(0, 5),
                              _lcm_cpu(21, 6), _lcm_cpu(-4, 6), _lcm_cpu(-6, -8)],
                             dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_int64():
    a = torch.tensor([10000000000, 999999999], dtype=torch.int64).cuda()
    b = torch.tensor([5000000000, 123456789], dtype=torch.int64).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([_lcm_cpu(10000000000, 5000000000),
                              _lcm_cpu(999999999, 123456789)], dtype=torch.int64).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_zero():
    # Test: lcm(a, 0) = 0 and lcm(0, b) = 0
    a = torch.tensor([42, 0, 0], dtype=torch.int32).cuda()
    b = torch.tensor([0, 100, 0], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([0, 0, 0], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_coprime():
    # Coprime numbers have LCM = product
    a = torch.tensor([7, 13, 17], dtype=torch.int32).cuda()
    b = torch.tensor([11, 17, 19], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([77, 221, 323], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_same_value():
    # lcm(a, a) = a
    a = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()
    b = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_2d():
    a = torch.tensor([[4, 6], [0, 21]], dtype=torch.int32).cuda()
    b = torch.tensor([[6, 8], [5, 6]], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([[12, 24], [0, 42]], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_lcm_negative():
    # LCM should work with negative numbers (uses absolute values)
    a = torch.tensor([-12, -15, 12], dtype=torch.int32).cuda()
    b = torch.tensor([18, -20, -18], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.lcm(a, b)

    expected = torch.tensor([36, 60, 36], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)

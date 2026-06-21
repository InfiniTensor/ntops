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


@skip_if_cuda_not_available
def test_gcd_int32():
    a = torch.tensor([48, 17, 0, 100, -48, -17, -100], dtype=torch.int32).cuda()
    b = torch.tensor([18, 13, 5, 0, 18, -13, -25], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.gcd(a, b)

    expected = torch.tensor([_gcd_cpu(48, 18), _gcd_cpu(17, 13), _gcd_cpu(0, 5),
                              _gcd_cpu(100, 0), _gcd_cpu(-48, 18), _gcd_cpu(-17, -13),
                              _gcd_cpu(-100, -25)], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_gcd_int64():
    a = torch.tensor([123456789012, 999999999999], dtype=torch.int64).cuda()
    b = torch.tensor([987654321098, 123456789012], dtype=torch.int64).cuda()

    ninetoothed_output = ntops.torch.gcd(a, b)

    expected = torch.tensor([_gcd_cpu(123456789012, 987654321098),
                              _gcd_cpu(999999999999, 123456789012)], dtype=torch.int64).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_gcd_fibonacci():
    # Test with consecutive Fibonacci numbers (worst case for Euclidean algorithm)
    # F(47) = 2971215073, F(46) = 1836311903
    a = torch.tensor([2971215073], dtype=torch.int64).cuda()
    b = torch.tensor([1836311903], dtype=torch.int64).cuda()

    ninetoothed_output = ntops.torch.gcd(a, b)

    # Consecutive Fibonacci numbers have GCD = 1
    expected = torch.tensor([1], dtype=torch.int64).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_gcd_same_value():
    a = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()
    b = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.gcd(a, b)

    expected = torch.tensor([42, 100, 0], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)


@skip_if_cuda_not_available
def test_gcd_2d():
    a = torch.tensor([[48, 17], [0, 100]], dtype=torch.int32).cuda()
    b = torch.tensor([[18, 13], [5, 0]], dtype=torch.int32).cuda()

    ninetoothed_output = ntops.torch.gcd(a, b)

    expected = torch.tensor([[6, 1], [5, 100]], dtype=torch.int32).cuda()

    assert torch.equal(ninetoothed_output, expected)

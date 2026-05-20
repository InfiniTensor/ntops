import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 64


def broadcast_2d_arrangement(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.expand((-1, other.shape[1]))
    other = other.expand((input.shape[0], -1))
    return tuple(tensor.flatten().tile((block_size,)) for tensor in (input, other, output))


def _gcd_parts(input, other, iterations):
    x = ntl.abs(input)
    y = ntl.abs(other)
    a = x
    b = y

    for _ in range(iterations):
        safe_b = ntl.where(b == 0, 1, b)
        r = a % safe_b
        a = ntl.where(b == 0, a, b)
        b = ntl.where(b == 0, b, r)

    return x, y, a


def _apply_lcm(input, other, output, iterations):
    x, y, gcd = _gcd_parts(input, other, iterations)
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    input_min = (input < 0) & (-input == input)
    other_min = (other < 0) & (-other == other)
    min_overflow = input_min | other_min
    overflow_value = ntl.where(input_min, input, other)
    value = ntl.where(min_overflow, overflow_value, value)
    output = ntl.where(gcd == 0, 0, value)  # noqa: F841


def _apply_lcm_abs(input, other, output, iterations):
    x, y, gcd = _gcd_parts(input, other, iterations)
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    value = ntl.abs((x // safe_gcd) * y)
    input_min = (input < 0) & (-input == input)
    other_min = (other < 0) & (-other == other)
    min_overflow = input_min | other_min
    overflow_value = ntl.where(input_min, input, other)
    value = ntl.where(min_overflow, overflow_value, value)
    output = ntl.where(gcd == 0, 0, value)  # noqa: F841


def _apply_lcm_dynamic(input, other, output, max_iterations, absolute_output):
    x = ntl.abs(input)
    y = ntl.abs(other)
    input_min = (input < 0) & (-input == input)
    other_min = (other < 0) & (-other == other)
    min_overflow = input_min | other_min
    a = ntl.where(min_overflow, 1, x)
    b = ntl.where(min_overflow, 1, y)
    iteration = 0

    while (ntl.max(b) != 0) and (iteration < max_iterations):
        safe_b = ntl.where(b == 0, 1, b)
        r = a % safe_b
        a = ntl.where(b == 0, a, b)
        b = ntl.where(b == 0, b, r)
        iteration += 1

    safe_gcd = ntl.where(a == 0, 1, a)
    value = (x // safe_gcd) * y
    if absolute_output:
        value = ntl.abs(value)
    overflow_value = ntl.where(input_min, input, other)
    value = ntl.where(min_overflow, overflow_value, value)
    output = ntl.where((input == 0) | (other == 0), 0, value)  # noqa: F841


def application_16(input, other, output):
    _apply_lcm(input, other, output, 16)


def application_16_dynamic(input, other, output):
    _apply_lcm_dynamic(input, other, output, 16, False)


def application_16_dynamic_i32(input, other, output):
    _apply_lcm_dynamic(ntl.cast(input, ntl.int32), ntl.cast(other, ntl.int32), output, 16, False)


def application_24(input, other, output):
    _apply_lcm(input, other, output, 24)


def application_24_dynamic(input, other, output):
    _apply_lcm_dynamic(input, other, output, 24, False)


def application_24_dynamic_i32(input, other, output):
    _apply_lcm_dynamic(ntl.cast(input, ntl.int32), ntl.cast(other, ntl.int32), output, 24, False)


def application_32(input, other, output):
    _apply_lcm(input, other, output, 32)


def application_48(input, other, output):
    _apply_lcm(input, other, output, 48)


def application_48_dynamic_abs(input, other, output):
    _apply_lcm_dynamic(input, other, output, 48, True)


def application_48_dynamic_i32(input, other, output):
    _apply_lcm_dynamic(ntl.cast(input, ntl.int32), ntl.cast(other, ntl.int32), output, 48, False)


def application_48_abs(input, other, output):
    _apply_lcm_abs(input, other, output, 48)


def application_64(input, other, output):
    _apply_lcm(input, other, output, 64)


def application_96(input, other, output):
    _apply_lcm(input, other, output, 96)


def application_96_dynamic_abs(input, other, output):
    _apply_lcm_dynamic(input, other, output, 96, True)


def application_96_abs(input, other, output):
    _apply_lcm_abs(input, other, output, 96)


def premake(
    ndim,
    iterations=96,
    absolute_output=False,
    dynamic_iterations=False,
    small_integer=False,
    broadcast_2d=False,
    dtype=None,
    block_size=BLOCK_SIZE,
):
    arrangement_func = broadcast_2d_arrangement if broadcast_2d else arrangement
    arrangement_ = functools.partial(arrangement_func, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    applications = {
        16: application_16,
        (16, False, True): application_16_dynamic,
        (16, False, True, True): application_16_dynamic_i32,
        24: application_24,
        (24, False, True): application_24_dynamic,
        (24, False, True, True): application_24_dynamic_i32,
        32: application_32,
        48: application_48,
        (48, True): application_48_abs,
        (48, True, True): application_48_dynamic_abs,
        (48, False, True, True): application_48_dynamic_i32,
        64: application_64,
        96: application_96,
        (96, True): application_96_abs,
        (96, True, True): application_96_dynamic_abs,
    }

    key = (
        (iterations, absolute_output, True, True)
        if dynamic_iterations and small_integer
        else (
            (iterations, absolute_output, True)
            if dynamic_iterations
            else ((iterations, True) if absolute_output else iterations)
        )
    )
    return arrangement_, applications[key], tensors

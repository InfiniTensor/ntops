import functools
import math

import torch
import triton
import triton.language as tl


def is_iluvatar_device(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda" or not hasattr(torch, "cuda"):
        return False
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    try:
        return "Iluvatar" in torch.cuda.get_device_name(index)
    except Exception:
        return False


@functools.cache
def _lcm_gcd_table(device_index):
    values = [math.gcd(lhs, rhs) for lhs in range(256) for rhs in range(256)]
    return torch.tensor(values, dtype=torch.int16, device=torch.device("cuda", device_index))


@functools.cache
def _lcm_u8_table(device_index):
    values = []
    for lhs in range(256):
        for rhs in range(256):
            values.append(0 if lhs == 0 or rhs == 0 else math.lcm(lhs, rhs) & 0xFF)
    return torch.tensor(values, dtype=torch.uint8, device=torch.device("cuda", device_index))


def lcm_gcd_table(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _lcm_gcd_table(index)


def lcm_u8_table(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _lcm_u8_table(index)


@triton.jit
def _rad2deg_f32_kernel(input, output, n: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    value = tl.load(input + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, value * 57.29577951308232, mask=mask)


@triton.jit
def _copysign_f32_broadcast_kernel(
    input,
    other,
    output,
    cols: tl.constexpr,
    n: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0.0)
    other_value = tl.load(other + col, mask=mask, other=0.0)
    input_bits = input_value.to(tl.uint32, bitcast=True)
    other_bits = other_value.to(tl.uint32, bitcast=True)
    output_bits = (input_bits & 0x7FFFFFFF) | (other_bits & 0x80000000)
    tl.store(output + offsets, output_bits.to(tl.float32, bitcast=True), mask=mask)


@triton.jit
def _nextafter_f32_broadcast_kernel(
    input,
    other,
    output,
    cols: tl.constexpr,
    n: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0.0)
    other_value = tl.load(other + col, mask=mask, other=0.0)

    bits = input_value.to(tl.uint32, bitcast=True)
    increment = tl.where(input_value > 0, other_value > input_value, other_value < input_value)
    next_bits = tl.where(increment, bits + 1, bits - 1)
    value = next_bits.to(tl.float32, bitcast=True)

    zero_value = tl.where(other_value < 0, -1.401298464324817e-45, 1.401298464324817e-45)
    zero_value = tl.where(other_value == 0, other_value, zero_value)
    value = tl.where(input_value == 0, zero_value, value)
    value = tl.where(input_value == other_value, other_value, value)
    value = tl.where(input_value != input_value, input_value, value)
    value = tl.where(other_value != other_value, other_value, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _nextafter_f16_broadcast_kernel(
    input,
    other,
    output,
    cols: tl.constexpr,
    n: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0.0)
    other_value = tl.load(other + col, mask=mask, other=0.0)

    bits = input_value.to(tl.uint16, bitcast=True)
    increment = tl.where(input_value > 0, other_value > input_value, other_value < input_value)
    next_bits = tl.where(increment, bits + 1, bits - 1).to(tl.uint16)
    value = next_bits.to(tl.float16, bitcast=True)

    zero_value = tl.where(other_value < 0, -5.960464477539063e-08, 5.960464477539063e-08)
    zero_value = zero_value.to(tl.float16)
    zero_value = tl.where(other_value == 0, other_value, zero_value)
    value = tl.where(input_value == 0, zero_value, value)
    value = tl.where(input_value == other_value, other_value, value)
    value = tl.where(input_value != input_value, input_value, value)
    value = tl.where(other_value != other_value, other_value, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _nextafter_f16_kernel(input, other, output, n: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    input_value = tl.load(input + offsets, mask=mask, other=0.0)
    other_value = tl.load(other + offsets, mask=mask, other=0.0)

    bits = input_value.to(tl.uint16, bitcast=True)
    increment = tl.where(input_value > 0, other_value > input_value, other_value < input_value)
    next_bits = tl.where(increment, bits + 1, bits - 1).to(tl.uint16)
    value = next_bits.to(tl.float16, bitcast=True)

    zero_value = tl.where(other_value < 0, -5.960464477539063e-08, 5.960464477539063e-08)
    zero_value = zero_value.to(tl.float16)
    zero_value = tl.where(other_value == 0, other_value, zero_value)
    value = tl.where(input_value == 0, zero_value, value)
    value = tl.where(input_value == other_value, other_value, value)
    value = tl.where(input_value != input_value, input_value, value)
    value = tl.where(other_value != other_value, other_value, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _nextafter_f16_strided_kernel(
    input,
    other,
    output,
    n: tl.constexpr,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    d4: tl.constexpr,
    d5: tl.constexpr,
    input_s0: tl.constexpr,
    input_s1: tl.constexpr,
    input_s2: tl.constexpr,
    input_s3: tl.constexpr,
    input_s4: tl.constexpr,
    input_s5: tl.constexpr,
    other_s0: tl.constexpr,
    other_s1: tl.constexpr,
    other_s2: tl.constexpr,
    other_s3: tl.constexpr,
    other_s4: tl.constexpr,
    other_s5: tl.constexpr,
    output_s0: tl.constexpr,
    output_s1: tl.constexpr,
    output_s2: tl.constexpr,
    output_s3: tl.constexpr,
    output_s4: tl.constexpr,
    output_s5: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n

    rem = offsets
    i5 = rem % d5
    rem = rem // d5
    i4 = rem % d4
    rem = rem // d4
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1

    input_offsets = (
        i0 * input_s0
        + i1 * input_s1
        + i2 * input_s2
        + i3 * input_s3
        + i4 * input_s4
        + i5 * input_s5
    )
    other_offsets = (
        i0 * other_s0
        + i1 * other_s1
        + i2 * other_s2
        + i3 * other_s3
        + i4 * other_s4
        + i5 * other_s5
    )
    output_offsets = (
        i0 * output_s0
        + i1 * output_s1
        + i2 * output_s2
        + i3 * output_s3
        + i4 * output_s4
        + i5 * output_s5
    )

    input_value = tl.load(input + input_offsets, mask=mask, other=0.0)
    other_value = tl.load(other + other_offsets, mask=mask, other=0.0)

    bits = input_value.to(tl.uint16, bitcast=True)
    increment = tl.where(input_value > 0, other_value > input_value, other_value < input_value)
    next_bits = tl.where(increment, bits + 1, bits - 1).to(tl.uint16)
    value = next_bits.to(tl.float16, bitcast=True)

    zero_value = tl.where(other_value < 0, -5.960464477539063e-08, 5.960464477539063e-08)
    zero_value = zero_value.to(tl.float16)
    zero_value = tl.where(other_value == 0, other_value, zero_value)
    value = tl.where(input_value == 0, zero_value, value)
    value = tl.where(input_value == other_value, other_value, value)
    value = tl.where(input_value != input_value, input_value, value)
    value = tl.where(other_value != other_value, other_value, value)
    tl.store(output + output_offsets, value, mask=mask)


@triton.jit
def _lcm_small_or_dynamic_kernel(
    input,
    other,
    output,
    gcd_table,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    output_bits: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    input_value = tl.load(input + offsets, mask=mask, other=0).to(tl.int64)
    other_value = tl.load(other + offsets, mask=mask, other=0).to(tl.int64)

    x = tl.abs(input_value)
    y = tl.abs(other_value)
    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int64)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < max_iterations):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output and output_bits == 32:
        value = value.to(tl.int32).to(tl.int64)
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_i32_small_or_dynamic_kernel(
    input,
    other,
    output,
    gcd_table,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    input_value = tl.load(input + offsets, mask=mask, other=0).to(tl.int32)
    other_value = tl.load(other + offsets, mask=mask, other=0).to(tl.int32)

    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min
    x = tl.abs(input_value)
    y = tl.abs(other_value)

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int32)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < max_iterations):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_i32_small_or_fixed_large_kernel(
    input,
    other,
    output,
    gcd_table,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    input_value = tl.load(input + offsets, mask=mask, other=0).to(tl.int32)
    other_value = tl.load(other + offsets, mask=mask, other=0).to(tl.int32)

    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min
    x = tl.abs(input_value)
    y = tl.abs(other_value)

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int32)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    if tl.max(tl.where(mask & (~small) & (~min_overflow), 1, 0), axis=0) != 0:
        for _ in range(max_iterations):
            safe_b = tl.where(b == 0, 1, b)
            r = a % safe_b
            a = tl.where(b == 0, a, b)
            b = tl.where(b == 0, b, r)

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_small_or_dynamic_broadcast_kernel(
    input,
    other,
    output,
    gcd_table,
    cols: tl.constexpr,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    output_bits: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0).to(tl.int64)
    other_value = tl.load(other + col, mask=mask, other=0).to(tl.int64)

    x = tl.abs(input_value)
    y = tl.abs(other_value)
    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int64)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < max_iterations):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output and output_bits == 32:
        value = value.to(tl.int32).to(tl.int64)
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_i32_small_or_dynamic_broadcast_kernel(
    input,
    other,
    output,
    gcd_table,
    cols: tl.constexpr,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0).to(tl.int32)
    other_value = tl.load(other + col, mask=mask, other=0).to(tl.int32)

    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min
    x = tl.abs(input_value)
    y = tl.abs(other_value)

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int32)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < max_iterations):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_i32_small_or_fixed_large_broadcast_kernel(
    input,
    other,
    output,
    gcd_table,
    cols: tl.constexpr,
    n: tl.constexpr,
    max_iterations: tl.constexpr,
    absolute_output: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    row = offsets // cols
    col = offsets - row * cols
    input_value = tl.load(input + row, mask=mask, other=0).to(tl.int32)
    other_value = tl.load(other + col, mask=mask, other=0).to(tl.int32)

    input_min = (input_value < 0) & (-input_value == input_value)
    other_min = (other_value < 0) & (-other_value == other_value)
    min_overflow = input_min | other_min
    x = tl.abs(input_value)
    y = tl.abs(other_value)

    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_index = x * 256 + y
    table_gcd = tl.load(gcd_table + table_index, mask=mask & small, other=0).to(tl.int32)

    a = tl.where(small | min_overflow, 0, x)
    b = tl.where(small | min_overflow, 0, y)
    if tl.max(tl.where(mask & (~small) & (~min_overflow), 1, 0), axis=0) != 0:
        for _ in range(max_iterations):
            safe_b = tl.where(b == 0, 1, b)
            r = a % safe_b
            a = tl.where(b == 0, a, b)
            b = tl.where(b == 0, b, r)

    gcd = tl.where(small, table_gcd, a)
    safe_gcd = tl.where(gcd == 0, 1, gcd)
    value = (x // safe_gcd) * y
    if absolute_output:
        value = tl.abs(value)
    overflow_value = tl.where(input_min, input_value, other_value)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((input_value == 0) | (other_value == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_u8_table_kernel(input, other, output, table, n: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    mask = offsets < n
    input_value = tl.load(input + offsets, mask=mask, other=0).to(tl.uint32)
    other_value = tl.load(other + offsets, mask=mask, other=0).to(tl.uint32)
    table_index = input_value * 256 + other_value
    value = tl.load(table + table_index, mask=mask, other=0)
    tl.store(output + offsets, value, mask=mask)


def copysign_f32_broadcast(input, other, output):
    rows = input.shape[0]
    cols = other.shape[1]
    n = rows * cols
    block = 256
    grid = (triton.cdiv(n, block),)
    _copysign_f32_broadcast_kernel[grid](
        input,
        other,
        output,
        cols,
        n,
        block=block,
        num_warps=4,
    )


def rad2deg_f32_1d(input, output):
    n = input.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _rad2deg_f32_kernel[grid](
        input,
        output,
        n,
        block=block,
        num_warps=4,
    )


def nextafter_f32_broadcast(input, other, output):
    rows = input.shape[0]
    cols = other.shape[1]
    n = rows * cols
    block = 256
    grid = (triton.cdiv(n, block),)
    _nextafter_f32_broadcast_kernel[grid](
        input,
        other,
        output,
        cols,
        n,
        block=block,
        num_warps=4,
    )


def nextafter_f16_broadcast(input, other, output):
    rows = input.shape[0]
    cols = other.shape[1]
    n = rows * cols
    block = 256
    grid = (triton.cdiv(n, block),)
    _nextafter_f16_broadcast_kernel[grid](
        input,
        other,
        output,
        cols,
        n,
        block=block,
        num_warps=4,
    )


def nextafter_f16_1d(input, other, output):
    n = input.numel()
    block = 512
    grid = (triton.cdiv(n, block),)
    _nextafter_f16_kernel[grid](
        input,
        other,
        output,
        n,
        block=block,
        num_warps=4,
    )


def _shape_and_strides_6d(tensor):
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    pad = 6 - len(shape)
    return (1,) * pad + shape, (0,) * pad + strides


def nextafter_f16_strided(input, other, output):
    if input.ndim > 6:
        return False
    n = output.numel()
    block = 512
    grid = (triton.cdiv(n, block),)
    shape, input_strides = _shape_and_strides_6d(input)
    _, other_strides = _shape_and_strides_6d(other)
    _, output_strides = _shape_and_strides_6d(output)
    _nextafter_f16_strided_kernel[grid](
        input,
        other,
        output,
        n,
        *shape,
        *input_strides,
        *other_strides,
        *output_strides,
        block=block,
        num_warps=4,
    )
    return True


def lcm_1d(input, other, output, max_iterations, absolute_output):
    n = input.numel()
    block = 128 if output.element_size() >= 4 else 256
    grid = (triton.cdiv(n, block),)
    if output.element_size() <= 2:
        _lcm_i32_small_or_fixed_large_kernel[grid](
            input,
            other,
            output,
            lcm_gcd_table(input.device),
            n,
            max_iterations,
            absolute_output,
            block=block,
            num_warps=1,
        )
        return
    if output.element_size() <= 4:
        _lcm_i32_small_or_dynamic_kernel[grid](
            input,
            other,
            output,
            lcm_gcd_table(input.device),
            n,
            max_iterations,
            absolute_output,
            block=block,
            num_warps=1,
        )
        return
    _lcm_small_or_dynamic_kernel[grid](
        input,
        other,
        output,
        lcm_gcd_table(input.device),
        n,
        max_iterations,
        absolute_output,
        output.element_size() * 8,
        block=block,
        num_warps=1,
    )


def lcm_u8_1d(input, other, output):
    n = input.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _lcm_u8_table_kernel[grid](
        input,
        other,
        output,
        lcm_u8_table(input.device),
        n,
        block=block,
        num_warps=4,
    )


def lcm_broadcast(input, other, output, max_iterations, absolute_output):
    rows = input.shape[0]
    cols = other.shape[1]
    n = rows * cols
    block = 128 if output.element_size() >= 4 else 256
    grid = (triton.cdiv(n, block),)
    if output.element_size() <= 2:
        _lcm_i32_small_or_fixed_large_broadcast_kernel[grid](
            input,
            other,
            output,
            lcm_gcd_table(input.device),
            cols,
            n,
            max_iterations,
            absolute_output,
            block=block,
            num_warps=1,
        )
        return
    if output.element_size() <= 4:
        _lcm_i32_small_or_dynamic_broadcast_kernel[grid](
            input,
            other,
            output,
            lcm_gcd_table(input.device),
            cols,
            n,
            max_iterations,
            absolute_output,
            block=block,
            num_warps=1,
        )
        return
    _lcm_small_or_dynamic_broadcast_kernel[grid](
        input,
        other,
        output,
        lcm_gcd_table(input.device),
        cols,
        n,
        max_iterations,
        absolute_output,
        output.element_size() * 8,
        block=block,
        num_warps=1,
    )

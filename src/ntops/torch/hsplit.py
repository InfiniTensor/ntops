# src/ntops/torch/hsplit.py

import torch

import ntops
from ntops.torch.utils import _cached_make


def _hsplit_dim(input):
    if input.ndim == 0:
        raise RuntimeError("hsplit expects a tensor with at least 1 dimension")

    return 0 if input.ndim == 1 else 1


def _unwrap_indices_or_sections(indices_or_sections):
    value = indices_or_sections

    # 兼容 infinicore 测试框架里的 Sections 包装类
    for name in (
        "value",
        "values",
        "data",
        "sections",
        "indices",
        "indices_or_sections",
    ):
        if hasattr(value, name):
            attr = getattr(value, name)
            if not callable(attr):
                value = attr
                break

    if not isinstance(value, (int, list, tuple)) and hasattr(value, "__dict__"):
        for attr in vars(value).values():
            if isinstance(attr, (int, list, tuple)):
                value = attr
                break

    return value


def _normalize_index(index, size):
    index = int(index)

    if index < 0:
        index += size

    return max(0, min(index, size))


def _get_split_ranges(size, indices_or_sections):
    indices_or_sections = _unwrap_indices_or_sections(indices_or_sections)

    if isinstance(indices_or_sections, int):
        sections = indices_or_sections

        if sections <= 0:
            raise RuntimeError("number of sections must be larger than 0")

        base = size // sections
        extra = size % sections

        ranges = []
        start = 0

        for i in range(sections):
            length = base + (1 if i < extra else 0)
            end = start + length
            ranges.append((start, end))
            start = end

        return ranges

    indices = [_normalize_index(index, size) for index in indices_or_sections]

    starts = [0] + indices
    ends = indices + [size]

    return list(zip(starts, ends))


def _empty_output(input, dim, length):
    output_shape = list(input.shape)
    output_shape[dim] = length

    return torch.empty(
        tuple(output_shape),
        dtype=input.dtype,
        device=input.device,
    )


def _copy_slice(input, dim, start, end):
    length = end - start

    output = _empty_output(input, dim, length)

    # 空切片不需要 launch kernel
    if length == 0:
        return output

    kernel = _cached_make(
        ntops.kernels.hsplit.premake,
        input.ndim,
        dim,
        start,
        end,
    )

    kernel(input, output)

    return output


def hsplit(input, indices_or_sections):
    dim = _hsplit_dim(input)
    size = input.shape[dim]

    ranges = _get_split_ranges(size, indices_or_sections)

    outputs = []

    for start, end in ranges:
        # fast path:
        # 如果这一段就是完整 input，直接返回 input，不 copy
        if start == 0 and end == size:
            outputs.append(input)
            continue

        outputs.append(_copy_slice(input, dim, start, end))

    return tuple(outputs)
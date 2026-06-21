import torch

import ntops
from ntops.torch.utils import _cached_make


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim

    assert 0 <= dim < ndim

    return dim


def _normalize_index(index, dim_size):
    if index < 0:
        index += dim_size

    if index < 0:
        index = 0

    if index > dim_size:
        index = dim_size

    return index


def _split_starts_and_sizes(dim_size, indices_or_sections):
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections

        assert sections > 0

        base = dim_size // sections
        extra = dim_size % sections

        sizes = []
        for i in range(sections):
            if i < extra:
                sizes.append(base + 1)
            else:
                sizes.append(base)

        starts = []
        start = 0
        for size in sizes:
            starts.append(start)
            start += size

        return starts, sizes

    if isinstance(indices_or_sections, torch.Tensor):
        indices = indices_or_sections.detach().cpu().tolist()
    else:
        indices = list(indices_or_sections)

    indices = [_normalize_index(int(index), dim_size) for index in indices]

    starts = [0] + indices
    ends = indices + [dim_size]

    sizes = []
    for start, end in zip(starts, ends):
        size = end - start
        if size < 0:
            size = 0
        sizes.append(size)

    return starts, sizes


def tensor_split(input, indices_or_sections, dim=0):
    ndim = input.ndim

    assert ndim > 0

    dim = _normalize_dim(dim, ndim)

    dim_size = input.shape[dim]
    starts, sizes = _split_starts_and_sizes(dim_size, indices_or_sections)

    kernel = _cached_make(ntops.kernels.tensor_split.premake, input.ndim)

    outputs = []

    for start, size in zip(starts, sizes):
        output_shape = list(input.shape)
        output_shape[dim] = size

        output = torch.empty(
            output_shape,
            dtype=input.dtype,
            device=input.device,
        )

        if output.numel() != 0:
            input_slice = input.narrow(dim, start, size)
            kernel(input_slice, output)

        outputs.append(output)

    return tuple(outputs)
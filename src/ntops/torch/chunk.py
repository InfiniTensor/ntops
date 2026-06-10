import torch

import ntops
from ntops.torch.utils import _cached_make


def chunk(input, chunks, dim=0):
    if dim < 0:
        dim = input.ndim + dim

    chunk_size = (input.shape[dim] + chunks - 1) // chunks

    outputs = []

    for i in range(chunks):
        start = i * chunk_size
        end = min(start + chunk_size, input.shape[dim])

        if start >= input.shape[dim]:
            break

        slices = [slice(None)] * input.ndim
        slices[dim] = slice(start, end)

        chunk_tensor = input[tuple(slices)]
        out_chunk = torch.empty_like(chunk_tensor)

        kernel = _cached_make(ntops.kernels.chunk.premake, input.ndim)
        kernel(chunk_tensor, out_chunk)

        outputs.append(out_chunk)

    return tuple(outputs)

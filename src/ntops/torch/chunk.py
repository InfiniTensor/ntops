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

        if start >= input.shape[dim]:
            break

        actual_size = min(chunk_size, input.shape[dim] - start)

        out_shape = list(input.shape)
        out_shape[dim] = actual_size
        out_chunk = torch.empty(out_shape, dtype=input.dtype, device=input.device)

        kernel = _cached_make(
            ntops.kernels.chunk.premake, input.ndim, dim, start, actual_size
        )
        kernel(input, out_chunk)

        outputs.append(out_chunk)

    return tuple(outputs)

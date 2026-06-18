import torch

import ninetoothed
import ntops
from ntops.torch.utils import _cached_make

_DTYPE_MAP = {
    torch.float16: ninetoothed.float16,
    torch.bfloat16: ninetoothed.bfloat16,
    torch.float32: ninetoothed.float32,
    torch.float64: ninetoothed.float64,
    torch.int8: ninetoothed.int8,
    torch.int16: ninetoothed.int16,
    torch.int32: ninetoothed.int32,
    torch.int64: ninetoothed.int64,
}


def chunk(input, chunks, dim=0):
    if dim < 0:
        dim = input.ndim + dim

    dim_size = input.shape[dim]
    chunk_size = (dim_size + chunks - 1) // chunks

    # Fast path: contiguous input — every narrow() along any dim produces a
    # contiguous view when the tensor is contiguous (dim=0) or when the sliced
    # dim is the leading dimension of a contiguous tensor.  For the most common
    # case (dim=0, contiguous input) all slices are contiguous, so we can
    # return views directly with zero kernel launches.
    if input.is_contiguous() and dim == 0:
        return tuple(
            input.narrow(0, i * chunk_size, min(chunk_size, dim_size - i * chunk_size))
            for i in range(chunks)
            if i * chunk_size < dim_size
        )

    # General path: slice in Python then decide per-chunk whether a kernel
    # copy is needed.  All chunks share one compiled kernel (cache key is
    # (premake, ndim, dtype) only — dim/start/size are no longer part of it).
    kernel = _cached_make(
        ntops.kernels.chunk.premake,
        input.ndim,
        dtype=_DTYPE_MAP.get(input.dtype),
    )

    outputs = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= dim_size:
            break

        actual_size = min(chunk_size, dim_size - start)
        chunk_view = input.narrow(dim, start, actual_size)

        if chunk_view.is_contiguous():
            outputs.append(chunk_view)
        else:
            out_chunk = torch.empty(
                chunk_view.shape, dtype=input.dtype, device=input.device
            )
            kernel(chunk_view, out_chunk)
            outputs.append(out_chunk)

    return tuple(outputs)

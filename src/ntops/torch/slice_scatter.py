import torch

from ntops.torch.copy import _copy


def slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    # `slice_scatter` returns a *new* tensor equal to `input` everywhere except
    # the slice `input[..., start:end:step, ...]` (along `dim`), which is taken
    # from `src`. The dominant cost is copying all of `input`; that contiguous
    # copy is the ninetoothed kernel. Writing `src` into the strided slice view
    # is a small torch op (glue), matching the `corrcoef`/`matmul` convention of
    # keeping the heavy, regular work on ninetoothed.
    input = input.contiguous()
    output = torch.empty_like(input)

    _copy(input, output)

    dim = dim % input.ndim
    index = [slice(None)] * input.ndim
    index[dim] = slice(start, end, step)
    output[tuple(index)] = src

    return output

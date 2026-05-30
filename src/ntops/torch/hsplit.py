import torch

from ntops.torch.copy import _copy


def hsplit(input, indices_or_sections):
    # `torch.hsplit` splits along dim 0 for 1D inputs and dim 1 otherwise, and
    # returns zero-copy *views*. ninetoothed cannot return views, so each split
    # is materialized into a contiguous tensor by a copy kernel reading the
    # (generally strided) slice. We compute the split boundaries ourselves
    # rather than calling `torch.hsplit`, so the operator is genuinely
    # reimplemented; only the cheap view construction borrows torch slicing.
    assert input.ndim >= 1, "`hsplit` requires at least a 1D tensor."

    dim = 0 if input.ndim == 1 else 1
    size = input.shape[dim]

    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        assert size % sections == 0, (
            f"torch.hsplit attempted to split along dimension {dim}, but the "
            f"size of the dimension {size} is not divisible by the split_size "
            f"{sections}."
        )
        split = size // sections
        bounds = [(i * split, (i + 1) * split) for i in range(sections)]
    else:
        points = [0, *list(indices_or_sections), size]
        bounds = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    outputs = []

    for lo, hi in bounds:
        index = [slice(None)] * input.ndim
        index[dim] = slice(lo, hi)
        view = input[tuple(index)]

        out = torch.empty(view.shape, dtype=input.dtype, device=input.device)

        if view.numel() != 0:
            _copy(view, out)

        outputs.append(out)

    return tuple(outputs)

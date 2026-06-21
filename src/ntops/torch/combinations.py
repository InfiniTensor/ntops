import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def _num_combinations(n, r, with_replacement):
    if r < 0:
        raise ValueError("r must be non-negative")

    if r == 0:
        return 1

    if n == 0:
        return 0

    if with_replacement:
        return math.comb(n + r - 1, r)

    if r > n:
        return 0

    return math.comb(n, r)


def combinations(input, r=2, with_replacement=False, *, out=None):
    assert input.ndim == 1, "combinations only supports 1-D input"

    r = int(r)
    with_replacement = bool(with_replacement)

    assert r >= 0, "r must be non-negative"
    assert r <= 3, "combinations currently only supports r <= 3"

    input_size = input.shape[0]

    num_rows = _num_combinations(
        n=input_size,
        r=r,
        with_replacement=with_replacement,
    )

    if out is None:
        out = torch.empty(
            (num_rows, r),
            dtype=input.dtype,
            device=input.device,
        )
    else:
        assert tuple(out.shape) == (num_rows, r), (
            f"invalid out shape, expected {(num_rows, r)}, got {tuple(out.shape)}"
        )
        assert out.dtype == input.dtype, "out dtype must match input dtype"
        assert out.device == input.device, "out device must match input device"

    if out.numel() == 0:
        return out

    kernel = _cached_make(
        ntops.kernels.combinations.premake,
        input_size,
        r,
        with_replacement,
    )

    kernel(input, out, input_size, r, with_replacement)

    return out
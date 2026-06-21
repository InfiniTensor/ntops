import torch

import ntops
from ntops.torch.utils import _cached_make


def _storage_ptr(x):
    candidates = [x]

    underlying = getattr(x, "_underlying", None)
    if underlying is not None:
        candidates.append(underlying)

    for obj in candidates:
        data_ptr = getattr(obj, "data_ptr", None)
        if callable(data_ptr):
            return data_ptr()

        if isinstance(obj, torch.Tensor):
            if hasattr(obj, "untyped_storage"):
                return obj.untyped_storage().data_ptr()

            return obj.storage().data_ptr()

    return None


def _same_storage(a, b):
    if a is b:
        return True

    pa = _storage_ptr(a)
    pb = _storage_ptr(b)

    return pa is not None and pb is not None and pa == pb


def heaviside(input, values, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.heaviside.premake,
        input.ndim,
        dtype=input.dtype,
    )

    need_tmp = _same_storage(out, input)

    actual_out = torch.empty_like(input) if need_tmp else out

    kernel(input, values, actual_out)

    if need_tmp:
        out.copy_(actual_out)

    return out

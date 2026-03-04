import torch

import ntops
from ntops.torch.utils import _cached_make, _pad_dims_to_next_power_of_2


def quantile(input, q, dim=None, keepdim=False, interpolation="linear", out=None):
    is_scalar = False

    if isinstance(q, float):
        q = torch.tensor([q], dtype=input.dtype, device=input.device)
        # Use `from_list` method to create a tensor in infinicore.
        # q = torch.from_list([q], dtype=input.dtype, device=input.device)
        is_scalar = True
    elif q.ndim == 0:
        q = q.unsqueeze(0)

    # If `dim` is None, `input` will be flattened before computation.
    ndim = None

    if dim is None:
        ndim = input.ndim
        # `flatten` is not supported in `infinicore.tensor`, use `view` instead.
        flattened_size = 1

        for s in input.shape:
            flattened_size *= s

        input = input.contiguous().view([flattened_size])
        dim = 0

    # Pad the `input` and `q` to the next power of 2 along the specified dimensions.
    input_padded = _pad_dims_to_next_power_of_2(input, dim, value=float("inf"))
    q_padded = _pad_dims_to_next_power_of_2(q, 0)

    copy_back = False

    if out is None:
        out_shape = list(input.shape)
        out_shape[dim] = 1

        if not keepdim:
            out_shape.pop(dim)
        elif ndim is not None:
            out_shape.extend([1] * (ndim - 1))

        out_shape.insert(0, q.shape[0])
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        if not out.is_contiguous():
            # Non-contiguous tensor does not work right,
            # so we create a new contiguous tensor and copy back the result after computation.
            original_out = out
            copy_back = True
            out = out.contiguous()

        if is_scalar:
            # If `q` is a scalar, the corresponding `output` will also be a scalar,
            # but the application uses `gather` to get the sorted values, which requires
            # the `output` to have at least 1 dimension. We can unsqueeze the `output`
            # to make it compatible with the application.
            out = out.unsqueeze(0)

    if keepdim:
        out_adjust = out.squeeze(dim + 1)
    else:
        out_adjust = out

    kernel = _cached_make(
        ntops.kernels.quantile.premake, input.ndim, out_adjust.ndim, dim, interpolation
    )

    kernel(input_padded, q_padded, input.shape[dim], out_adjust)

    if is_scalar:
        out = out.squeeze(0)

    if copy_back:
        original_out.copy_(out)
        out = original_out

    return out

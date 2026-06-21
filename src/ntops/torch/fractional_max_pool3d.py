import torch

import ntops
from ntops.torch.utils import _cached_make


_CAST_EXCEPTIONS = (TypeError, RuntimeError, AttributeError, NotImplementedError)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)

    return value


def _calculate_fractional_output_size(
    input_size,
    output_size=None,
    output_ratio=None,
):
    assert output_size is not None or output_ratio is not None, (
        "Either `output_size` or `output_ratio` must be specified."
    )
    assert output_size is None or output_ratio is None, (
        "`output_size` and `output_ratio` cannot both be specified."
    )

    if output_size is not None:
        return output_size

    return int(input_size * output_ratio)


def _empty_like_input(input, shape):
    empty_fn = getattr(torch, "empty", None)

    if callable(empty_fn):
        try:
            return empty_fn(
                shape,
                dtype=input.dtype,
                device=input.device,
            )
        except _CAST_EXCEPTIONS:
            return _empty_like_input_by_zeros(input, shape)

    return _empty_like_input_by_zeros(input, shape)


def _empty_like_input_by_zeros(input, shape):
    zeros_fn = getattr(torch, "zeros", None)

    if callable(zeros_fn):
        try:
            return zeros_fn(
                shape,
                dtype=input.dtype,
                device=input.device,
            )
        except _CAST_EXCEPTIONS as exc:
            raise RuntimeError(
                "Cannot create output tensor for fractional_max_pool3d."
            ) from exc

    raise RuntimeError(
        "Cannot create output tensor for fractional_max_pool3d."
    )


def _make_fractional_sequence(
    input_size,
    output_size,
    kernel_size,
    random_samples,
):
    if output_size == 1:
        return torch.full(
            random_samples.shape + (1,),
            input_size - kernel_size,
            dtype=torch.int64,
            device=random_samples.device,
        )

    alpha = float(input_size - kernel_size) / float(output_size - 1)

    index = torch.arange(
        output_size,
        dtype=torch.float32,
        device=random_samples.device,
    )

    sample = random_samples.float()

    sequence = torch.floor(
        (index + sample[..., None]) * alpha
    ) - torch.floor(
        sample[..., None] * alpha
    )

    sequence = sequence.long()
    sequence[..., -1] = input_size - kernel_size

    return sequence


def _fractional_max_pool3d_with_random_samples(
    input,
    kernel_size,
    d_out,
    h_out,
    w_out,
    _random_samples,
):
    n, c, d, h, w = input.shape

    assert _random_samples.shape == (n, c, 3), (
        "`_random_samples` must have shape `(N, C, 3)`."
    )

    # PyTorch fractional_max_pool3d:
    # samples[..., 0] -> D
    # samples[..., 1] -> H
    # samples[..., 2] -> W
    sequence_d_start = _make_fractional_sequence(
        d,
        d_out,
        kernel_size[0],
        _random_samples[..., 0],
    )
    sequence_h_start = _make_fractional_sequence(
        h,
        h_out,
        kernel_size[1],
        _random_samples[..., 1],
    )
    sequence_w_start = _make_fractional_sequence(
        w,
        w_out,
        kernel_size[2],
        _random_samples[..., 2],
    )

    sequence_d_end = sequence_d_start + kernel_size[0]
    sequence_h_end = sequence_h_start + kernel_size[1]
    sequence_w_end = sequence_w_start + kernel_size[2]

    sequence_d_start = sequence_d_start[:, :, :, None, None].expand(
        -1,
        -1,
        -1,
        h_out,
        w_out,
    ).contiguous()
    sequence_d_end = sequence_d_end[:, :, :, None, None].expand(
        -1,
        -1,
        -1,
        h_out,
        w_out,
    ).contiguous()

    sequence_h_start = sequence_h_start[:, :, None, :, None].expand(
        -1,
        -1,
        d_out,
        -1,
        w_out,
    ).contiguous()
    sequence_h_end = sequence_h_end[:, :, None, :, None].expand(
        -1,
        -1,
        d_out,
        -1,
        w_out,
    ).contiguous()

    sequence_w_start = sequence_w_start[:, :, None, None, :].expand(
        -1,
        -1,
        d_out,
        h_out,
        -1,
    ).contiguous()
    sequence_w_end = sequence_w_end[:, :, None, None, :].expand(
        -1,
        -1,
        d_out,
        h_out,
        -1,
    ).contiguous()

    output = _empty_like_input(
        input,
        (n, c, d_out, h_out, w_out),
    )

    kernel = _cached_make(
        ntops.kernels.fractional_max_pool3d.premake,
        block_size=1,
        input_d_upper_bound=d,
        input_h_upper_bound=h,
        input_w_upper_bound=w,
        output_d_upper_bound=d_out,
        output_h_upper_bound=h_out,
        output_w_upper_bound=w_out,
    )

    kernel(
        input,
        sequence_d_start,
        sequence_d_end,
        sequence_h_start,
        sequence_h_end,
        sequence_w_start,
        sequence_w_end,
        output,
    )

    return output


def _fractional_max_pool3d_deterministic(
    input,
    kernel_size,
    d_out,
    h_out,
    w_out,
):
    n, c, d, h, w = input.shape

    output = _empty_like_input(
        input,
        (n, c, d_out, h_out, w_out),
    )

    kernel = _cached_make(
        ntops.kernels.fractional_max_pool3d.premake_deterministic,
        block_size=1,
        input_d_upper_bound=d,
        input_h_upper_bound=h,
        input_w_upper_bound=w,
        output_d_upper_bound=d_out,
        output_h_upper_bound=h_out,
        output_w_upper_bound=w_out,
        kernel_d=kernel_size[0],
        kernel_h=kernel_size[1],
        kernel_w=kernel_size[2],
    )

    kernel(
        input,
        d,
        h,
        w,
        d_out,
        h_out,
        w_out,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        output,
    )

    return output


def fractional_max_pool3d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
    _random_samples=None,
):
    assert input.ndim == 5, "`fractional_max_pool3d` only supports 5D input for now."
    assert not return_indices, "`return_indices` is not supported yet."

    kernel_size = _triple(kernel_size)

    if output_size is not None:
        output_size = _triple(output_size)

    if output_ratio is not None:
        output_ratio = _triple(output_ratio)

    n, c, d, h, w = input.shape

    d_out = _calculate_fractional_output_size(
        d,
        output_size=None if output_size is None else output_size[0],
        output_ratio=None if output_ratio is None else output_ratio[0],
    )
    h_out = _calculate_fractional_output_size(
        h,
        output_size=None if output_size is None else output_size[1],
        output_ratio=None if output_ratio is None else output_ratio[1],
    )
    w_out = _calculate_fractional_output_size(
        w,
        output_size=None if output_size is None else output_size[2],
        output_ratio=None if output_ratio is None else output_ratio[2],
    )

    assert d_out > 0 and h_out > 0 and w_out > 0, "`output_size` must be positive."

    assert d_out + kernel_size[0] - 1 <= d, (
        "`output_size[0] + kernel_size[0] - 1` must be no greater than input depth."
    )
    assert h_out + kernel_size[1] - 1 <= h, (
        "`output_size[1] + kernel_size[1] - 1` must be no greater than input height."
    )
    assert w_out + kernel_size[2] - 1 <= w, (
        "`output_size[2] + kernel_size[2] - 1` must be no greater than input width."
    )

    if _random_samples is not None:
        return _fractional_max_pool3d_with_random_samples(
            input,
            kernel_size,
            d_out,
            h_out,
            w_out,
            _random_samples,
        )

    return _fractional_max_pool3d_deterministic(
        input,
        kernel_size,
        d_out,
        h_out,
        w_out,
    )
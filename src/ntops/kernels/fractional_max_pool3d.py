import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor


def _arrange_output(output, block_size):
    arranged = output.tile((1, 1, 1, 1, 1))
    arranged = arranged.ravel()
    arranged = arranged.flatten(end_dim=5).flatten(start_dim=1)
    arranged = arranged.tile((block_size, -1))
    arranged.dtype = arranged.dtype.squeeze(1)

    return arranged


def _arrange_sequence(sequence, input, block_size):
    arranged = sequence.tile((1, 1, 1, 1, 1))
    arranged = arranged.ravel()
    arranged = arranged.flatten(end_dim=5).flatten(start_dim=1)
    arranged = arranged.expand(
        (-1, input.shape[-3] * input.shape[-2] * input.shape[-1])
    )
    arranged = arranged.tile((block_size, -1))

    return arranged


def arrangement(
    input,
    sequence_d_start,
    sequence_d_end,
    sequence_h_start,
    sequence_h_end,
    sequence_w_start,
    sequence_w_end,
    output,
    block_size=1,
):
    if block_size is None:
        block_size = 1

    input_arranged = input.tile((1, 1, -1, -1, -1))
    input_arranged = input_arranged.expand(
        (-1, -1, output.shape[-3], output.shape[-2], output.shape[-1])
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size, -1))

    sequence_d_start_arranged = _arrange_sequence(
        sequence_d_start,
        input,
        block_size,
    )
    sequence_d_end_arranged = _arrange_sequence(
        sequence_d_end,
        input,
        block_size,
    )
    sequence_h_start_arranged = _arrange_sequence(
        sequence_h_start,
        input,
        block_size,
    )
    sequence_h_end_arranged = _arrange_sequence(
        sequence_h_end,
        input,
        block_size,
    )
    sequence_w_start_arranged = _arrange_sequence(
        sequence_w_start,
        input,
        block_size,
    )
    sequence_w_end_arranged = _arrange_sequence(
        sequence_w_end,
        input,
        block_size,
    )

    output_arranged = _arrange_output(output, block_size)

    return (
        input_arranged,
        sequence_d_start_arranged,
        sequence_d_end_arranged,
        sequence_h_start_arranged,
        sequence_h_end_arranged,
        sequence_w_start_arranged,
        sequence_w_end_arranged,
        output_arranged,
    )


def application(
    input,
    sequence_d_start,
    sequence_d_end,
    sequence_h_start,
    sequence_h_end,
    sequence_w_start,
    sequence_w_end,
    output,
):
    d_offsets = input.offsets(2)
    h_offsets = input.offsets(3)
    w_offsets = input.offsets(4)

    mask = (
        (d_offsets >= sequence_d_start)
        & (d_offsets < sequence_d_end)
        & (h_offsets >= sequence_h_start)
        & (h_offsets < sequence_h_end)
        & (w_offsets >= sequence_w_start)
        & (w_offsets < sequence_w_end)
    )

    neg_large = input * 0 - 65504.0
    masked_input = ntl.where(mask, input, neg_large)

    output = ntl.max(masked_input, axis=-1)  # noqa: F841


def arrangement_deterministic(
    input,
    input_d,
    input_h,
    input_w,
    output_d,
    output_h,
    output_w,
    kernel_d,
    kernel_h,
    kernel_w,
    output,
    block_size=1,
):
    if block_size is None:
        block_size = 1

    input_arranged = input.tile((1, 1, -1, -1, -1))
    input_arranged = input_arranged.expand(
        (-1, -1, output.shape[-3], output.shape[-2], output.shape[-1])
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size, -1))

    output_arranged = _arrange_output(output, block_size)

    return (
        input_arranged,
        input_d,
        input_h,
        input_w,
        output_d,
        output_h,
        output_w,
        kernel_d,
        kernel_h,
        kernel_w,
        output_arranged,
    )


def application_deterministic(
    input,
    input_d,
    input_h,
    input_w,
    output_d,
    output_h,
    output_w,
    kernel_d,
    kernel_h,
    kernel_w,
    output,
):
    d_offsets = input.offsets(2)
    h_offsets = input.offsets(3)
    w_offsets = input.offsets(4)

    od_offsets = output.offsets(2)
    oh_offsets = output.offsets(3)
    ow_offsets = output.offsets(4)

    d_start = (
        od_offsets * (input_d - kernel_d)
    ) // (output_d - 1)
    d_start = ntl.where(
        od_offsets == output_d - 1,
        input_d - kernel_d,
        d_start,
    )

    h_start = (
        oh_offsets * (input_h - kernel_h)
    ) // (output_h - 1)
    h_start = ntl.where(
        oh_offsets == output_h - 1,
        input_h - kernel_h,
        h_start,
    )

    w_start = (
        ow_offsets * (input_w - kernel_w)
    ) // (output_w - 1)
    w_start = ntl.where(
        ow_offsets == output_w - 1,
        input_w - kernel_w,
        w_start,
    )

    d_end = d_start + kernel_d
    h_end = h_start + kernel_h
    w_end = w_start + kernel_w

    mask = (
        (d_offsets >= d_start)
        & (d_offsets < d_end)
        & (h_offsets >= h_start)
        & (h_offsets < h_end)
        & (w_offsets >= w_start)
        & (w_offsets < w_end)
    )

    neg_large = input * 0 - 65504.0
    masked_input = ntl.where(mask, input, neg_large)

    output = ntl.max(masked_input, axis=-1)  # noqa: F841


def _make_sequence_tensor(
    output_d_upper_bound,
    output_h_upper_bound,
    output_w_upper_bound,
):
    return Tensor(
        5,
        shape_options=(
            None,
            None,
            {"constexpr": True, "upper_bound": output_d_upper_bound},
            {"constexpr": True, "upper_bound": output_h_upper_bound},
            {"constexpr": True, "upper_bound": output_w_upper_bound},
        ),
    )


def premake(
    dtype=None,
    block_size=1,
    input_d_upper_bound=128,
    input_h_upper_bound=128,
    input_w_upper_bound=128,
    output_d_upper_bound=128,
    output_h_upper_bound=128,
    output_w_upper_bound=128,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    input = Tensor(
        5,
        dtype=dtype,
        other=-65504.0,
        shape_options=(
            None,
            None,
            {"constexpr": True, "upper_bound": input_d_upper_bound},
            {"constexpr": True, "upper_bound": input_h_upper_bound},
            {"constexpr": True, "upper_bound": input_w_upper_bound},
        ),
    )

    sequence_d_start = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )
    sequence_d_end = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )
    sequence_h_start = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )
    sequence_h_end = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )
    sequence_w_start = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )
    sequence_w_end = _make_sequence_tensor(
        output_d_upper_bound,
        output_h_upper_bound,
        output_w_upper_bound,
    )

    output = Tensor(
        5,
        dtype=dtype,
        shape_options=(
            None,
            None,
            {"constexpr": True, "upper_bound": output_d_upper_bound},
            {"constexpr": True, "upper_bound": output_h_upper_bound},
            {"constexpr": True, "upper_bound": output_w_upper_bound},
        ),
    )

    tensors = (
        input,
        sequence_d_start,
        sequence_d_end,
        sequence_h_start,
        sequence_h_end,
        sequence_w_start,
        sequence_w_end,
        output,
    )

    return arrangement_, application, tensors


def premake_deterministic(
    dtype=None,
    block_size=1,
    input_d_upper_bound=128,
    input_h_upper_bound=128,
    input_w_upper_bound=128,
    output_d_upper_bound=128,
    output_h_upper_bound=128,
    output_w_upper_bound=128,
    kernel_d=1,
    kernel_h=1,
    kernel_w=1,
):
    arrangement_ = functools.partial(
        arrangement_deterministic,
        block_size=block_size,
    )

    input = Tensor(
        5,
        dtype=dtype,
        other=-65504.0,
        shape_options=(
            None,
            None,
            {"constexpr": True, "upper_bound": input_d_upper_bound},
            {"constexpr": True, "upper_bound": input_h_upper_bound},
            {"constexpr": True, "upper_bound": input_w_upper_bound},
        ),
    )

    input_d_tensor = Tensor(0)
    input_h_tensor = Tensor(0)
    input_w_tensor = Tensor(0)
    output_d_tensor = Tensor(0)
    output_h_tensor = Tensor(0)
    output_w_tensor = Tensor(0)
    kernel_d_tensor = Tensor(0)
    kernel_h_tensor = Tensor(0)
    kernel_w_tensor = Tensor(0)

    output = Tensor(
        5,
        dtype=dtype,
        shape_options=(
            None,
            None,
            {"constexpr": True, "upper_bound": output_d_upper_bound},
            {"constexpr": True, "upper_bound": output_h_upper_bound},
            {"constexpr": True, "upper_bound": output_w_upper_bound},
        ),
    )

    tensors = (
        input,
        input_d_tensor,
        input_h_tensor,
        input_w_tensor,
        output_d_tensor,
        output_h_tensor,
        output_w_tensor,
        kernel_d_tensor,
        kernel_h_tensor,
        kernel_w_tensor,
        output,
    )

    return arrangement_, application_deterministic, tensors
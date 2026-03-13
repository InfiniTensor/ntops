import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    input,
    output,
    target_dim_size,
    in_dims=None,
    out_dims=None,
    axis=None,
    axis_is_none=False,
    target_dim_size_power2=None,
):
    if axis_is_none:
        input = input.flatten()
        input_arranged = input.tile((target_dim_size_power2,))
        if out_dims > 0:
            output = output.flatten()
            output_arranged = output.tile((-1,))
        else:
            output_arranged = output.unsqueeze(dim=0)
            output_arranged = output_arranged.tile((1,))
    else:
        if in_dims == 4:
            if axis == 0:
                input_arranged = input.tile((target_dim_size_power2, -1, -1, -1))
            elif axis == 1:
                input_arranged = input.tile((-1, target_dim_size_power2, -1, -1))
            elif axis == 2:
                input_arranged = input.tile((-1, -1, target_dim_size_power2, -1))
            elif axis == 3:
                input_arranged = input.tile((-1, -1, -1, target_dim_size_power2))
            input_arranged = input_arranged.squeeze(0)
            input_arranged = input_arranged.squeeze(0)
            input_arranged = input_arranged.squeeze(0)
        elif in_dims == 3:
            if axis == 0:
                input_arranged = input.tile((target_dim_size_power2, -1, -1))
            elif axis == 1:
                input_arranged = input.tile((-1, target_dim_size_power2, -1))
            elif axis == 2:
                input_arranged = input.tile((-1, -1, target_dim_size_power2))
            input_arranged = input_arranged.squeeze(0)
            input_arranged = input_arranged.squeeze(0)
        elif in_dims == 2:
            if axis == 0:
                input_arranged = input.tile((target_dim_size_power2, -1))
            elif axis == 1:
                input_arranged = input.tile((-1, target_dim_size_power2))
            input_arranged = input_arranged.squeeze(0)
        elif in_dims == 1:
            input_arranged = input.tile((target_dim_size_power2,))

        if out_dims == 4:
            output_arranged = output.tile((-1, -1, -1, -1))
            output_arranged = output_arranged.squeeze(0)
            output_arranged = output_arranged.squeeze(0)
            output_arranged = output_arranged.squeeze(0)
        elif out_dims == 3:
            output_arranged = output.tile((-1, -1, -1))
            output_arranged = output_arranged.squeeze(0)
            output_arranged = output_arranged.squeeze(0)
        elif out_dims == 2:
            output_arranged = output.tile((-1, -1))
            output_arranged = output_arranged.squeeze(0)
        elif out_dims == 1:
            output_arranged = output.tile((-1,))
        elif out_dims == 0:
            output_arranged = output.unsqueeze(dim=0)
            output_arranged = output_arranged.tile((1,))

    return input_arranged, output_arranged, target_dim_size


def application_0_1(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[0]) < target_dim_size
    if input.ndim == 1:
        mask = valid_mask
    elif input.ndim == 2:
        mask = valid_mask[:, None]
    elif input.ndim == 3:
        mask = valid_mask[:, None, None]
    elif input.ndim == 4:
        mask = valid_mask[:, None, None, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=0, keep_dims=1)


def application_1_1(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[1]) < target_dim_size
    if input.ndim == 2:
        mask = valid_mask[None, :]
    elif input.ndim == 3:
        mask = valid_mask[None, :, None]
    elif input.ndim == 4:
        mask = valid_mask[None, :, None, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=1, keep_dims=1)


def application_2_1(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[2]) < target_dim_size
    if input.ndim == 3:
        mask = valid_mask[None, None, :]
    elif input.ndim == 4:
        mask = valid_mask[None, None, :, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=2, keep_dims=1)


def application_3_1(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[3]) < target_dim_size
    if input.ndim == 4:
        mask = valid_mask[None, None, None, :]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=3, keep_dims=1)


def application_0_0(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[0]) < target_dim_size
    if input.ndim == 1:
        mask = valid_mask
    elif input.ndim == 2:
        mask = valid_mask[:, None]
    elif input.ndim == 3:
        mask = valid_mask[:, None, None]
    elif input.ndim == 4:
        mask = valid_mask[:, None, None, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=0, keep_dims=0)


def application_1_0(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[1]) < target_dim_size
    if input.ndim == 2:
        mask = valid_mask[None, :]
    elif input.ndim == 3:
        mask = valid_mask[None, :, None]
    elif input.ndim == 4:
        mask = valid_mask[None, :, None, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=1, keep_dims=0)


def application_2_0(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[2]) < target_dim_size
    if input.ndim == 3:
        mask = valid_mask[None, None, :]
    elif input.ndim == 4:
        mask = valid_mask[None, None, :, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=2, keep_dims=0)


def application_3_0(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[3]) < target_dim_size
    if input.ndim == 4:
        mask = valid_mask[None, None, None, :]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    output = ntl.argmin(masked_input, axis=3, keep_dims=0)


def application_0_0_scalar(input, output, target_dim_size):
    valid_mask = ntl.arange(0, input.shape[0]) < target_dim_size
    if input.ndim == 1:
        mask = valid_mask
    elif input.ndim == 2:
        mask = valid_mask[:, None]
    elif input.ndim == 3:
        mask = valid_mask[:, None, None]
    elif input.ndim == 4:
        mask = valid_mask[:, None, None, None]
    else:
        raise ValueError(f"Unsupported input ndim: {input.ndim}")
    masked_input = ntl.where(mask, input, float("inf"))
    result = ntl.argmin(masked_input, axis=0, keep_dims=0)
    output[0] = result


def premake(
    target_dim_size,
    dtype=None,
    in_dims=None,
    out_dims=None,
    axis=None,
    axis_is_none=False,
    keep_dims=None,
):
    import math

    target_dim_size_power2 = (
        2 ** math.ceil(math.log2(int(target_dim_size)))
        if int(target_dim_size) > 0
        else 1
    )
    arrangement_ = functools.partial(
        arrangement,
        in_dims=in_dims,
        out_dims=out_dims,
        axis=axis,
        axis_is_none=axis_is_none,
        target_dim_size_power2=target_dim_size_power2,
    )

    tensors = (
        Tensor(in_dims, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(out_dims, dtype=ninetoothed.int64, shape_options={"constexpr": True}),
        Tensor(0, constexpr=True, value=target_dim_size),
    )
    # if out_dims > 0:
    #     tensors = (
    #         Tensor(in_dims, dtype=dtype, shape_options={"constexpr": True}),
    #         Tensor(out_dims, dtype=ninetoothed.int64, shape_options={"constexpr": True}),
    #     )
    # else:
    #     # 标量输出：不使用 constexpr
    #     tensors = (
    #         Tensor(in_dims, dtype=dtype, shape_options={"constexpr": True}),
    #         Tensor(0, dtype=ninetoothed.int64),  # ← 删除 shape_options
    #     )

    if axis_is_none:
        if keep_dims:
            application = application_0_1
        else:
            application = application_0_0
    else:
        if axis == 0:
            if keep_dims:
                application = application_0_1
            else:
                application = application_0_0
        elif axis == 1:
            if keep_dims:
                application = application_1_1
            else:
                application = application_1_0
        elif axis == 2:
            if keep_dims:
                application = application_2_1
            else:
                application = application_2_0
        elif axis == 3:
            if keep_dims:
                application = application_3_1
            else:
                application = application_3_0
        else:
            raise ValueError(f"Unsupported axis: {axis}")

    if out_dims == 0:
        application = application_0_0_scalar

    return arrangement_, application, tensors

import functools

import ninetoothed
from ninetoothed import Symbol, Tensor


def application(input, output):
    output = input  # noqa: F841


def arrangement(
    input,
    output,
    downscale_factor=None,
    block_size=None,
):
    if downscale_factor is None:
        downscale_factor = Symbol(
            "downscale_factor",
            constexpr=True,
            upper_bound=16,
        )

    if block_size is None:
        block_size = ninetoothed.block_size()

    factor2 = downscale_factor * downscale_factor

    # input: [N, C, H * r, W * r]
    # arranged: [N, C, H, W, r * r]
    input_arranged = input.tile(
        (1, 1, downscale_factor, downscale_factor),
        strides=(-1, -1, downscale_factor, downscale_factor),
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((block_size, -1))

    # output: [N, C * r * r, H, W]
    # arranged: [N, C, H, W, r * r]
    output_arranged = output.tile(
        (1, factor2, 1, 1),
        strides=(-1, factor2, -1, -1),
    )
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    return input_arranged, output_arranged


def premake(
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(
            4,
            dtype=dtype,
            shape_options=(
                None,
                None,
                None,
                {"constexpr": True, "upper_bound": 8192},
            ),
        ),
        Tensor(
            4,
            dtype=dtype,
            shape_options=(
                None,
                None,
                None,
                {"constexpr": True, "upper_bound": 8192},
            ),
        ),
    )

    return arrangement_, application, tensors
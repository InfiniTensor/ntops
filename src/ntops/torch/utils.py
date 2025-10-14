import functools

import ninetoothed
import torch

import ntops


@functools.cache
def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE

    return ntops.kernels.mm.InputPrecisionVariant.TF32

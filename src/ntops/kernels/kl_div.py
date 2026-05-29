import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as _element_wise_arrangement


# ---------------------------------------------------------------------------
# Element-wise path (reduction="none"): output keeps the input shape.
#
# Pointwise KL divergence (matching torch.nn.functional.kl_div):
#   * log_target=False: target * (log(target) - input), with the
#     0 * log(0) = 0 convention -> contributions where target <= 0 are zeroed
#     via ``where`` (this also masks the log(0) = -inf produced by padding).
#   * log_target=True:  exp(target) * (target - input).
# The math runs in float32 for accuracy (log/exp lose precision in fp16); the
# store casts back to the output dtype, mirroring the silu/softmax kernels.
# ---------------------------------------------------------------------------


def application(input, target, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    target_fp32 = ntl.cast(target, ntl.float32)
    pointwise = target_fp32 * (ntl.log(target_fp32) - input_fp32)
    output = ntl.where(target_fp32 > 0, pointwise, 0)  # noqa: F841


def log_target_application(input, target, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    target_fp32 = ntl.cast(target, ntl.float32)
    output = ntl.exp(target_fp32) * (target_fp32 - input_fp32)  # noqa: F841


def premake(ndim, log_target=False, dtype=None, block_size=None):
    arrangement_ = functools.partial(_element_wise_arrangement, block_size=block_size)

    application_ = log_target_application if log_target else application

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application_, tensors


# ---------------------------------------------------------------------------
# Reduction path (reduction="sum"/"mean"/"batchmean"): a single-pass partial-sum
# kernel emits one float32 partial per block; the host sums the partials and
# applies the reduction scaling. ~2N memory traffic, no full intermediate
# tensor. ``other=0`` pads the trailing block: target=0 yields a zeroed
# contribution under both formulas, so padding never perturbs the sum.
# ---------------------------------------------------------------------------


def _reduce_arrangement(input, target, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.flatten().tile((block_size,))
    target_arranged = target.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((1,))

    return input_arranged, target_arranged, output_arranged


def reduce_application(input, target, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    target_fp32 = ntl.cast(target, ntl.float32)
    pointwise = target_fp32 * (ntl.log(target_fp32) - input_fp32)
    masked = ntl.where(target_fp32 > 0, pointwise, 0)
    output = ntl.sum(masked)  # noqa: F841


def reduce_log_target_application(input, target, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    target_fp32 = ntl.cast(target, ntl.float32)
    output = ntl.sum(ntl.exp(target_fp32) * (target_fp32 - input_fp32))  # noqa: F841


def reduce_premake(input_dtype=None, log_target=False, block_size=None):
    arrangement_ = functools.partial(_reduce_arrangement, block_size=block_size)

    application_ = (
        reduce_log_target_application if log_target else reduce_application
    )

    tensors = (
        Tensor(1, other=0, dtype=input_dtype),
        Tensor(1, other=0, dtype=input_dtype),
        Tensor(1, dtype=ninetoothed.float32),
    )

    return arrangement_, application_, tensors

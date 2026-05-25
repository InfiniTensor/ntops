import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, tau, hard, output):
    out_dtype = output.dtype.dtype

    tau_f = ntl.cast(tau, ntl.float32)
    hard_enabled = hard != ntl.cast(0.0, ntl.float64)

    zero_f = ntl.cast(0.0, ntl.float32)
    one_f = ntl.cast(1.0, ntl.float32)
    eps_f = ntl.cast(1.0e-6, ntl.float32)
    neg_inf_f = ntl.cast(float("-inf"), ntl.float32)

    prev_max = neg_inf_f
    denominator = zero_f

    # First pass: compute max and denominator for stable softmax.
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)

        # For masked lanes, input_i may be -inf. Do not feed -inf to sin.
        seed_input_i = ntl.where(input_i == neg_inf_f, zero_f, input_i)

        idx_f = ntl.cast(i + 1, ntl.float32)

        seed_i = (
            seed_input_i * ntl.cast(12.9898, ntl.float32)
            + idx_f * ntl.cast(78.233, ntl.float32)
        )

        random_i = ntl.sin(seed_i) * ntl.cast(43758.5453, ntl.float32)
        random_floor_i = ntl.floor(random_i)
        u_raw_i = random_i - random_floor_i

        u_min_i = ntl.maximum(u_raw_i, eps_f)
        u_i = ntl.where(u_min_i > one_f - eps_f, one_f - eps_f, u_min_i)

        log_u_i = ntl.log(u_i)
        neg_log_u_i = -log_u_i
        log_neg_log_u_i = ntl.log(neg_log_u_i)
        gumbel_i = -log_neg_log_u_i

        value_i = (input_i + gumbel_i) / tau_f

        block_max_i = ntl.max(value_i)
        curr_max = ntl.cast(ntl.maximum(prev_max, block_max_i), ntl.float32)

        value_diff_i = value_i - curr_max
        value_exp_i = ntl.exp(value_diff_i)

        prev_diff_i = prev_max - curr_max
        prev_exp_i = ntl.exp(prev_diff_i)

        denominator = denominator * prev_exp_i + ntl.sum(value_exp_i)
        prev_max = curr_max

    # Second pass: write soft or hard output.
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)

        seed_input_i = ntl.where(input_i == neg_inf_f, zero_f, input_i)

        idx_f = ntl.cast(i + 1, ntl.float32)

        seed_i = (
            seed_input_i * ntl.cast(12.9898, ntl.float32)
            + idx_f * ntl.cast(78.233, ntl.float32)
        )

        random_i = ntl.sin(seed_i) * ntl.cast(43758.5453, ntl.float32)
        random_floor_i = ntl.floor(random_i)
        u_raw_i = random_i - random_floor_i

        u_min_i = ntl.maximum(u_raw_i, eps_f)
        u_i = ntl.where(u_min_i > one_f - eps_f, one_f - eps_f, u_min_i)

        log_u_i = ntl.log(u_i)
        neg_log_u_i = -log_u_i
        log_neg_log_u_i = ntl.log(neg_log_u_i)
        gumbel_i = -log_neg_log_u_i

        value_i = (input_i + gumbel_i) / tau_f

        soft_exp_i = ntl.exp(value_i - prev_max)
        soft_i = soft_exp_i / denominator

        hard_i = ntl.where(value_i == prev_max, one_f, zero_f)

        result_i = ntl.where(hard_enabled, hard_i, soft_i)

        output[i] = ntl.cast(result_i, out_dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(
            ndim,
            dtype=dtype,
            other=float("-inf"),
            shape_options={"constexpr": True},
        ),
        Tensor(0, dtype=ninetoothed.float64),  # tau
        Tensor(0, dtype=ninetoothed.float64),  # hard: 0.0 / 1.0
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
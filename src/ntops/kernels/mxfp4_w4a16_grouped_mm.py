import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=16)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=16)
MICROSCALE_K = 32
PACKED_MICROSCALE_K = MICROSCALE_K // 2


def _arrange_activation(mat_a, output_arranged, block_size_m):
    arranged = mat_a.tile(
        (1, block_size_m, PACKED_MICROSCALE_K),
        strides=(1, block_size_m, MICROSCALE_K),
        dilation=(1, 1, 2),
    )
    arranged = arranged.tile((1, 1, -1))
    arranged = arranged.expand((-1, -1, output_arranged.shape[-1]))
    arranged.dtype = arranged.dtype.squeeze((0, 1))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def arrangement(
    mat_a_even,
    mat_a_odd,
    mat_b,
    scale_b,
    output,
    block_size_m=None,
    block_size_n=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    mat_a_even_arranged = _arrange_activation(mat_a_even, output_arranged, block_size_m)
    mat_a_odd_arranged = _arrange_activation(mat_a_odd, output_arranged, block_size_m)

    mat_b_arranged = mat_b.tile((1, PACKED_MICROSCALE_K, block_size_n))
    mat_b_arranged = mat_b_arranged.tile((1, -1, 1))
    mat_b_arranged = mat_b_arranged.expand((-1, output_arranged.shape[-2], -1))
    mat_b_arranged.dtype = mat_b_arranged.dtype.squeeze((0, 2))
    mat_b_arranged.dtype.dtype = mat_b_arranged.dtype.dtype.squeeze(0)

    scale_b_arranged = scale_b.tile((1, 1, block_size_n))
    scale_b_arranged = scale_b_arranged.tile((1, -1, 1))
    scale_b_arranged = scale_b_arranged.expand((-1, output_arranged.shape[-2], -1))
    scale_b_arranged.dtype = scale_b_arranged.dtype.squeeze((0, 2))
    scale_b_arranged.dtype.dtype = scale_b_arranged.dtype.dtype.squeeze(0)

    return (
        mat_a_even_arranged,
        mat_a_odd_arranged,
        mat_b_arranged,
        scale_b_arranged,
        output_arranged,
    )


def mxfp4_w4a16_grouped_mm_dot(mat_a_even, mat_a_odd, mat_b, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(mat_a_even.shape[0]):
        packed = (mat_b[k] + 0).to(ntl.int32)
        scale = ntl.exp2((scale_b[k] + 0).to(ntl.float32) - 127.0)

        even_code = packed & 0xF
        even_magnitude_code = even_code & 0x7
        even_exponent = even_magnitude_code >> 1
        even_mantissa = ((even_magnitude_code & 1) + 0).to(ntl.float32)
        even_normal = (1.0 + 0.5 * even_mantissa) * ntl.exp2(
            (even_exponent + 0).to(ntl.float32) - 1.0
        )
        even_magnitude = ntl.where(even_exponent == 0, 0.5 * even_mantissa, even_normal)
        even_sign = ntl.where((even_code & 0x8) == 0, 1.0, -1.0)
        weight_even = (even_sign * even_magnitude * scale).to(ntl.bfloat16)

        odd_code = (packed >> 4) & 0xF
        odd_magnitude_code = odd_code & 0x7
        odd_exponent = odd_magnitude_code >> 1
        odd_mantissa = ((odd_magnitude_code & 1) + 0).to(ntl.float32)
        odd_normal = (1.0 + 0.5 * odd_mantissa) * ntl.exp2(
            (odd_exponent + 0).to(ntl.float32) - 1.0
        )
        odd_magnitude = ntl.where(odd_exponent == 0, 0.5 * odd_mantissa, odd_normal)
        odd_sign = ntl.where((odd_code & 0x8) == 0, 1.0, -1.0)
        weight_odd = (odd_sign * odd_magnitude * scale).to(ntl.bfloat16)

        accumulator += ntl.dot(mat_a_even[k], weight_even)
        accumulator += ntl.dot(mat_a_odd[k], weight_odd)

    output = accumulator


def mxfp4_w4a16_grouped_mm_reduction(mat_a_even, mat_a_odd, mat_b, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(mat_a_even.shape[0]):
        packed = (mat_b[k] + 0).to(ntl.int32)
        scale = ntl.exp2((scale_b[k] + 0).to(ntl.float32) - 127.0)

        even_code = packed & 0xF
        even_bit_0 = even_code & 1
        even_bit_1 = (even_code >> 1) & 1
        even_bit_2 = (even_code >> 2) & 1
        even_low_magnitude = even_bit_0 + (even_bit_1 << 1)
        even_twice_magnitude = even_low_magnitude + even_bit_2 * (
            4 + even_bit_0 + (even_bit_1 << 1) + ((even_bit_0 * even_bit_1) << 1)
        )
        even_sign = 1 - (((even_code >> 3) & 1) << 1)
        decoded_even = ((even_sign * even_twice_magnitude) + 0).to(ntl.float32) * 0.5
        weight_even = (decoded_even * scale).to(ntl.bfloat16)

        odd_code = (packed >> 4) & 0xF
        odd_bit_0 = odd_code & 1
        odd_bit_1 = (odd_code >> 1) & 1
        odd_bit_2 = (odd_code >> 2) & 1
        odd_low_magnitude = odd_bit_0 + (odd_bit_1 << 1)
        odd_twice_magnitude = odd_low_magnitude + odd_bit_2 * (
            4 + odd_bit_0 + (odd_bit_1 << 1) + ((odd_bit_0 * odd_bit_1) << 1)
        )
        odd_sign = 1 - (((odd_code >> 3) & 1) << 1)
        decoded_odd = ((odd_sign * odd_twice_magnitude) + 0).to(ntl.float32) * 0.5
        weight_odd = (decoded_odd * scale).to(ntl.bfloat16)
        activation_even = (mat_a_even[k] + 0).to(ntl.float32)
        activation_odd = (mat_a_odd[k] + 0).to(ntl.float32)
        weight_even = (weight_even + 0).to(ntl.float32)
        weight_odd = (weight_odd + 0).to(ntl.float32)

        accumulator += ntl.sum(
            activation_even[:, :, None] * weight_even[None, :, :], axis=1
        )
        accumulator += ntl.sum(
            activation_odd[:, :, None] * weight_odd[None, :, :], axis=1
        )

    output = accumulator


def premake(jagged=False, block_size_m=None, block_size_n=None, reduction=False):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    jagged_dim = 1 if jagged else None
    # HIP uses the reduction path only. Specializing its matrix dimensions lets
    # Triton fold the repeated shape predicates before AMD LLVM codegen.
    dense_shape_options = {"constexpr": True} if reduction else None
    activation_shape_options = (
        ({"constexpr": True}, None, {"constexpr": True})
        if reduction and jagged
        else dense_shape_options
    )
    common_tensors = (
        Tensor(
            3,
            dtype=torch.bfloat16,
            jagged_dim=jagged_dim,
            other=0,
            shape_options=activation_shape_options,
        ),
        Tensor(
            3,
            dtype=torch.bfloat16,
            jagged_dim=jagged_dim,
            other=0,
            shape_options=activation_shape_options,
        ),
        Tensor(3, dtype=torch.uint8, other=0, shape_options=dense_shape_options),
        Tensor(3, dtype=torch.uint8, other=127, shape_options=dense_shape_options),
    )
    output_tensor = (
        Tensor(
            3,
            dtype=torch.bfloat16,
            jagged_dim=jagged_dim,
            shape_options=activation_shape_options,
        ),
    )
    tensors = common_tensors + output_tensor

    application = (
        mxfp4_w4a16_grouped_mm_reduction if reduction else mxfp4_w4a16_grouped_mm_dot
    )
    return arrangement_, application, tensors

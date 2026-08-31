import enum

import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make


class _ScalingType(enum.IntEnum):
    TensorWise = 0
    RowWise = 1
    BlockWise1x16 = 2
    BlockWise1x32 = 3
    BlockWise1x128 = 4
    BlockWise128x128 = 5


class _SwizzleType(enum.IntEnum):
    NO_SWIZZLE = 0
    SWIZZLE_32_4_4 = 1


ScalingType = getattr(F, "ScalingType", _ScalingType)
SwizzleType = getattr(F, "SwizzleType", _SwizzleType)
# Multi-wave launches regress the routed gfx936 specialization by more than an
# order of magnitude. Four waves still wins the uniform cases.
_HIP_AUTOTUNE_WARPS = (1, 4)


def _kernel_launch_config(jagged=False):
    if torch.version.hip is not None:
        if jagged:
            return True, 1, 16, 1, 1, 1
        return True, 1, 16, _HIP_AUTOTUNE_WARPS, 1, len(_HIP_AUTOTUNE_WARPS)
    return False, 16, 64, 4, 1, 1


def _make_kernel(jagged):
    reduction, block_size_m, block_size_n, num_warps, num_stages, limit = (
        _kernel_launch_config(jagged)
    )
    return _cached_make(
        ntops.kernels.mxfp4_w4a16_grouped_mm.premake,
        jagged,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        reduction=reduction,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=limit,
    )


def _dtype_if_available(name):
    return getattr(torch, name, None)


def _is_dtype(dtype, expected):
    return dtype is expected or (expected is not None and dtype == expected)


def _enum_matches(value, enum_type, member_name):
    member = getattr(enum_type, member_name)

    if value == member:
        return True

    return getattr(value, "name", None) == member_name


def _require_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_contiguous(name, value):
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _unwrap_single_level(name, value):
    if not isinstance(value, (tuple, list)):
        return value

    if len(value) != 1:
        raise NotImplementedError(f"{name} must contain exactly one level")

    return value[0]


def _as_uint8(value):
    if value.dtype == torch.uint8:
        return value

    return value.view(torch.uint8)


def _validate_optional_arguments(
    scale_a,
    scale_recipe_a,
    scale_recipe_b,
    swizzle_a,
    swizzle_b,
    bias,
    output_dtype,
    contraction_dim,
    use_fast_accum,
):
    if scale_a is not None:
        raise NotImplementedError("scale_a is not supported for W4A16")

    if scale_recipe_a is not None:
        raise NotImplementedError("scale_recipe_a is not supported for W4A16")

    if not _enum_matches(scale_recipe_b, ScalingType, "BlockWise1x32"):
        raise ValueError("scale_recipe_b must be ScalingType.BlockWise1x32")

    if swizzle_a is not None and not _enum_matches(
        swizzle_a, SwizzleType, "NO_SWIZZLE"
    ):
        raise NotImplementedError("swizzle_a is not supported")

    if swizzle_b is not None and not _enum_matches(
        swizzle_b, SwizzleType, "NO_SWIZZLE"
    ):
        raise NotImplementedError("swizzled inputs are not supported")

    if bias is not None:
        raise NotImplementedError("bias is not supported")

    if output_dtype not in (None, torch.bfloat16):
        raise ValueError("output_dtype must be torch.bfloat16")

    if contraction_dim not in (None, (), []):
        raise NotImplementedError("contraction_dim is not supported")

    if use_fast_accum:
        raise NotImplementedError("use_fast_accum=True is not supported")


def _validate_inputs(mat_a, mat_b, scale_b, offs):
    _require_tensor("mat_a", mat_a)
    _require_tensor("mat_b", mat_b)
    _require_tensor("scale_b", scale_b)

    if mat_a.dtype != torch.bfloat16:
        raise TypeError("mat_a must have dtype torch.bfloat16")

    packed_dtype = _dtype_if_available("float4_e2m1fn_x2")
    if mat_b.dtype != torch.uint8 and not _is_dtype(mat_b.dtype, packed_dtype):
        raise TypeError("mat_b must have dtype torch.uint8 or torch.float4_e2m1fn_x2")

    scale_dtype = _dtype_if_available("float8_e8m0fnu")
    if scale_b.dtype != torch.uint8 and not _is_dtype(scale_b.dtype, scale_dtype):
        raise TypeError("scale_b must have dtype torch.uint8 or torch.float8_e8m0fnu")

    if mat_b.ndim != 3:
        raise ValueError("mat_b must have shape (G, K // 2, N)")

    if scale_b.ndim != 3:
        raise ValueError("scale_b must have shape (G, K // 32, N)")

    _require_contiguous("mat_a", mat_a)
    _require_contiguous("mat_b", mat_b)
    _require_contiguous("scale_b", scale_b)

    group_count, packed_k, n = mat_b.shape
    if group_count == 0:
        raise ValueError("mat_b must contain at least one group")

    k = packed_k * 2
    if k == 0 or k % 32 != 0:
        raise ValueError("the logical K dimension must be a positive multiple of 32")

    if scale_b.shape != (group_count, k // 32, n):
        raise ValueError(
            f"scale_b must have shape {(group_count, k // 32, n)}, "
            f"but got {tuple(scale_b.shape)}"
        )

    device = mat_a.device
    if mat_b.device != device or scale_b.device != device:
        raise ValueError("mat_a, mat_b, and scale_b must be on the same device")

    if offs is None:
        if mat_a.ndim != 3:
            raise ValueError("mat_a must have shape (G, M, K) when offs is None")

        if mat_a.shape[0] != group_count or mat_a.shape[2] != k:
            raise ValueError(
                f"mat_a must have shape (G, M, K) with G={group_count} and K={k}"
            )

        return False, group_count, mat_a.shape[1], k, n

    _require_tensor("offs", offs)
    if mat_a.ndim != 2:
        raise ValueError("mat_a must have shape (total_M, K) when offs is provided")

    if mat_a.shape[1] != k:
        raise ValueError(f"mat_a must have K={k}, but got K={mat_a.shape[1]}")

    if offs.ndim != 1 or offs.numel() != group_count:
        raise ValueError(f"offs must have shape ({group_count},)")

    if offs.dtype != torch.int32:
        raise TypeError("offs must have dtype torch.int32")

    if offs.device != device:
        raise ValueError("offs must be on the same device as the inputs")

    _require_contiguous("offs", offs)

    if bool(torch.any(offs < 0).item()):
        raise ValueError("offs must contain non-negative cumulative row counts")

    if group_count > 1 and bool(torch.any(offs[1:] < offs[:-1]).item()):
        raise ValueError("offs must be nondecreasing")

    if int(offs[-1].item()) != mat_a.shape[0]:
        raise ValueError("offs[-1] must equal mat_a.shape[0]")

    return True, group_count, mat_a.shape[0], k, n


def mxfp4_w4a16_grouped_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    offs=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
):
    """Compute grouped MXFP4 W4A16 expert matrix multiplication."""
    scale_b = _unwrap_single_level("scale_b", scale_b)
    scale_recipe_b = _unwrap_single_level("scale_recipe_b", scale_recipe_b)
    swizzle_a = _unwrap_single_level("swizzle_a", swizzle_a)
    swizzle_b = _unwrap_single_level("swizzle_b", swizzle_b)

    _validate_optional_arguments(
        scale_a,
        scale_recipe_a,
        scale_recipe_b,
        swizzle_a,
        swizzle_b,
        bias,
        output_dtype,
        contraction_dim,
        use_fast_accum,
    )
    jagged, group_count, m, _, n = _validate_inputs(mat_a, mat_b, scale_b, offs)

    output_shape = (group_count, m, n) if not jagged else (m, n)
    output = torch.empty(output_shape, dtype=torch.bfloat16, device=mat_a.device)
    if output.numel() == 0:
        return output

    # The block-dot decoder lowers to BF16 MMAC on AMD and crashes gfx936
    # codegen. Use a branchless decoder with an ordinary FP32 reduction on HIP.
    kernel = _make_kernel(jagged)
    mat_b_uint8 = _as_uint8(mat_b)
    scale_b_uint8 = _as_uint8(scale_b)
    mat_a_even = mat_a[..., :-1]
    mat_a_odd = mat_a[..., 1:]

    common_args = (mat_a_even, mat_a_odd, mat_b_uint8, scale_b_uint8)

    if not jagged:
        kernel(*common_args, output)
        return output

    offsets = torch.cat((offs.new_zeros(1), offs))
    mat_a_even_jagged = torch.nested.nested_tensor_from_jagged(mat_a_even, offsets)
    mat_a_odd_jagged = torch.nested.nested_tensor_from_jagged(mat_a_odd, offsets)
    output_jagged = torch.nested.nested_tensor_from_jagged(output, offsets)
    jagged_args = (
        mat_a_even_jagged,
        mat_a_odd_jagged,
        *common_args[2:],
        output_jagged,
    )
    kernel(*jagged_args)

    return output

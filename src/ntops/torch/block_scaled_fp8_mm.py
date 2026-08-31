import torch

import ntops
from ntops.torch.mxfp4_w4a16_grouped_mm import ScalingType, SwizzleType
from ntops.torch.utils import _cached_make

_SCALE_BLOCK_SIZE = 128
_SUPPORTED_INPUT_DTYPE = torch.float8_e4m3fn


def _kernel_launch_config(device):
    if torch.version.hip is not None:
        return 32, 1

    is_corex = bool(getattr(torch, "corex", False))
    has_cuda = torch.version.cuda is not None and torch.cuda.is_available()
    if not is_corex and has_cuda:
        capability = torch.cuda.get_device_capability(device)
        # NVIDIA's native E4M3 Tensor Core path starts at SM89.
        if capability >= (8, 9):
            return 32, 4

    return 16, 4


def _kernel_tuning_config(device):
    block_size_k, num_warps = _kernel_launch_config(device)
    # gfx936 only compiles the one-wave specialization reliably. Multi-wave
    # candidates crash AMD make_amdgcn, which an in-process tuner cannot catch.
    return block_size_k, num_warps, 1, 1


def _make_kernel(input_dtype, output_dtype, bias_dtype, device):
    block_size_k, num_warps, num_stages, max_num_configs = _kernel_tuning_config(device)
    return _cached_make(
        ntops.kernels.block_scaled_fp8_mm.premake,
        input_dtype,
        output_dtype,
        bias_dtype,
        block_size_m=16,
        block_size_n=16,
        block_size_k=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _enum_matches(value, enum_type, member_name):
    member = getattr(enum_type, member_name)
    return value == member or getattr(value, "name", None) == member_name


def _require_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _unwrap_single_level(name, value):
    if not isinstance(value, (tuple, list)):
        return value
    if len(value) != 1:
        raise NotImplementedError(f"{name} must contain exactly one level")
    return value[0]


def _is_dense_matrix(value):
    return value.is_contiguous() or value.t().is_contiguous()


def _validate_options(
    scale_recipe_a,
    scale_recipe_b,
    swizzle_a,
    swizzle_b,
    output_dtype,
    contraction_dim,
    use_fast_accum,
):
    if not _enum_matches(scale_recipe_a, ScalingType, "BlockWise1x128"):
        raise NotImplementedError("scale_recipe_a must be ScalingType.BlockWise1x128")
    if not (
        _enum_matches(scale_recipe_b, ScalingType, "BlockWise1x128")
        or _enum_matches(scale_recipe_b, ScalingType, "BlockWise128x128")
    ):
        raise NotImplementedError(
            "scale_recipe_b must be ScalingType.BlockWise1x128 or "
            "ScalingType.BlockWise128x128"
        )
    for name, value in (("swizzle_a", swizzle_a), ("swizzle_b", swizzle_b)):
        if value is not None and not _enum_matches(value, SwizzleType, "NO_SWIZZLE"):
            raise NotImplementedError(f"{name} must be SwizzleType.NO_SWIZZLE")
    if output_dtype is None:
        output_dtype = torch.bfloat16
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "output_dtype must be torch.bfloat16, torch.float16, or torch.float32"
        )
    if contraction_dim not in (None, (), []):
        raise NotImplementedError("contraction_dim is not supported")
    if use_fast_accum:
        raise NotImplementedError("use_fast_accum=True is not supported")
    return output_dtype


def _validate_inputs(mat_a, mat_b, scale_a, scale_b, scale_recipe_b, bias):
    for name, value in (
        ("mat_a", mat_a),
        ("mat_b", mat_b),
        ("scale_a", scale_a),
        ("scale_b", scale_b),
    ):
        _require_tensor(name, value)

    if mat_a.dtype != _SUPPORTED_INPUT_DTYPE or mat_b.dtype != _SUPPORTED_INPUT_DTYPE:
        raise TypeError("mat_a and mat_b must have dtype torch.float8_e4m3fn")
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise TypeError("scale_a and scale_b must have dtype torch.float32")
    if mat_a.ndim != 2 or mat_b.ndim != 2:
        raise ValueError("mat_a and mat_b must both be 2D tensors")
    if scale_a.ndim != 2 or scale_b.ndim != 2:
        raise ValueError("scale_a and scale_b must both be 2D tensors")
    if not mat_a.is_contiguous():
        raise ValueError("mat_a must be row-major contiguous")
    if not mat_b.t().is_contiguous():
        raise ValueError("mat_b must be column-major (a transposed contiguous matrix)")
    if not _is_dense_matrix(scale_a) or not _is_dense_matrix(scale_b):
        raise ValueError(
            "scale_a and scale_b must have a dense row- or column-major layout"
        )

    m, k = mat_a.shape
    mat_b_k, n = mat_b.shape
    if mat_b_k != k:
        raise ValueError(
            f"mat_a.shape[1] must equal mat_b.shape[0], got {k} and {mat_b_k}"
        )
    if k == 0 or k % _SCALE_BLOCK_SIZE != 0:
        raise ValueError("K must be a positive multiple of 128")
    if n % _SCALE_BLOCK_SIZE != 0:
        raise ValueError("N must be a multiple of 128")

    k_blocks = k // _SCALE_BLOCK_SIZE
    n_blocks = n // _SCALE_BLOCK_SIZE
    if scale_a.shape != (m, k_blocks):
        raise ValueError(
            f"scale_a must have shape {(m, k_blocks)}, got {tuple(scale_a.shape)}"
        )

    b_uses_1x128 = _enum_matches(scale_recipe_b, ScalingType, "BlockWise1x128")
    if b_uses_1x128:
        expected_scale_b_shape = (n, k_blocks)
        if scale_b.shape != expected_scale_b_shape:
            raise ValueError(
                f"scale_b must have shape {expected_scale_b_shape}, "
                f"got {tuple(scale_b.shape)}"
            )
    else:
        padded_k_blocks = ((k_blocks + 3) // 4) * 4
        valid_shapes = ((k_blocks, n_blocks), (padded_k_blocks, n_blocks))
        if scale_b.shape not in valid_shapes:
            raise ValueError(
                "scale_b must have shape "
                f"{valid_shapes[0]} or padded shape {valid_shapes[1]}, "
                f"got {tuple(scale_b.shape)}"
            )

    device = mat_a.device
    if any(value.device != device for value in (mat_b, scale_a, scale_b)):
        raise ValueError(
            "mat_a, mat_b, scale_a, and scale_b must be on the same device"
        )

    if bias is not None:
        _require_tensor("bias", bias)
        if bias.device != device:
            raise ValueError("bias must be on the same device as the inputs")
        if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(
                "bias must have dtype torch.bfloat16, torch.float16, or torch.float32"
            )
        if bias.shape != (n,):
            raise ValueError(f"bias must have shape {(n,)}, got {tuple(bias.shape)}")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")

    return m, n, k_blocks, n_blocks, b_uses_1x128


def _group_views(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    output,
    bias,
    k_blocks,
    n_blocks,
    b_uses_1x128,
):
    m, k = mat_a.shape
    mat_a_groups = mat_a.unsqueeze(0).expand(n_blocks, -1, -1)
    mat_b_groups = mat_b.t().reshape(n_blocks, _SCALE_BLOCK_SIZE, k)
    mat_b_groups = mat_b_groups.transpose(-2, -1)
    scale_a_groups = scale_a.unsqueeze(0).expand(n_blocks, -1, -1)

    if b_uses_1x128:
        scale_b_groups = scale_b.reshape(n_blocks, _SCALE_BLOCK_SIZE, k_blocks).permute(
            0, 2, 1
        )
    else:
        scale_b_groups = scale_b[:k_blocks].t().unsqueeze(-1)
        scale_b_groups = scale_b_groups.expand(-1, -1, _SCALE_BLOCK_SIZE)

    output_groups = output.view(m, n_blocks, _SCALE_BLOCK_SIZE).permute(1, 0, 2)
    bias_groups = None
    if bias is not None:
        bias_groups = bias.view(n_blocks, 1, _SCALE_BLOCK_SIZE)

    return (
        mat_a_groups,
        mat_b_groups,
        scale_a_groups,
        scale_b_groups,
        bias_groups,
        output_groups,
    )


def block_scaled_fp8_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
):
    """Block-scaled FP8 matrix multiplication for LLM projections."""
    scale_a = _unwrap_single_level("scale_a", scale_a)
    scale_recipe_a = _unwrap_single_level("scale_recipe_a", scale_recipe_a)
    scale_b = _unwrap_single_level("scale_b", scale_b)
    scale_recipe_b = _unwrap_single_level("scale_recipe_b", scale_recipe_b)
    swizzle_a = _unwrap_single_level("swizzle_a", swizzle_a)
    swizzle_b = _unwrap_single_level("swizzle_b", swizzle_b)

    output_dtype = _validate_options(
        scale_recipe_a,
        scale_recipe_b,
        swizzle_a,
        swizzle_b,
        output_dtype,
        contraction_dim,
        use_fast_accum,
    )
    m, n, k_blocks, n_blocks, b_uses_1x128 = _validate_inputs(
        mat_a, mat_b, scale_a, scale_b, scale_recipe_b, bias
    )
    output = torch.empty((m, n), dtype=output_dtype, device=mat_a.device)
    if output.numel() == 0:
        return output

    grouped = _group_views(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        output,
        bias,
        k_blocks,
        n_blocks,
        b_uses_1x128,
    )
    (
        mat_a_groups,
        mat_b_groups,
        scale_a_groups,
        scale_b_groups,
        bias_groups,
        output_groups,
    ) = grouped

    kernel = _make_kernel(
        mat_a.dtype,
        output_dtype,
        bias.dtype if bias is not None else None,
        mat_a.device,
    )
    arguments = (mat_a_groups, mat_b_groups, scale_a_groups, scale_b_groups)
    if bias_groups is None:
        kernel(*arguments, output_groups)
    else:
        kernel(*arguments, bias_groups, output_groups)
    return output

import torch

import ntops
from ntops.torch.utils import _cached_make


_FP8_DTYPES = tuple(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _scaling_variant(input, mat2, scale_a, scale_b):
    m, k = input.shape
    n = mat2.shape[1]

    num_m_blocks = _ceil_div(m, ntops.kernels.scaled_mm.SCALE_BLOCK_SIZE)
    num_n_blocks = _ceil_div(n, ntops.kernels.scaled_mm.SCALE_BLOCK_SIZE)
    num_k_blocks = _ceil_div(k, ntops.kernels.scaled_mm.SCALE_BLOCK_SIZE)

    if scale_a.shape == (m, num_k_blocks):
        if scale_b.shape == (num_k_blocks, num_n_blocks):
            return ntops.kernels.scaled_mm.ScalingVariant.BLOCK_WISE_1X128_128X128

        if scale_b.shape == (num_k_blocks, n):
            return ntops.kernels.scaled_mm.ScalingVariant.BLOCK_WISE_1X128_1X128

    if scale_a.shape == (num_m_blocks, num_k_blocks) and scale_b.shape == (
        num_k_blocks,
        n,
    ):
        return ntops.kernels.scaled_mm.ScalingVariant.BLOCK_WISE_128X128_1X128

    raise ValueError(
        "Unsupported block scaling shapes. Expected one of "
        f"scale_a={(m, num_k_blocks)}, scale_b={(num_k_blocks, num_n_blocks)}; "
        f"scale_a={(m, num_k_blocks)}, scale_b={(num_k_blocks, n)}; or "
        f"scale_a={(num_m_blocks, num_k_blocks)}, scale_b={(num_k_blocks, n)}. "
        f"Got scale_a={tuple(scale_a.shape)} and scale_b={tuple(scale_b.shape)}."
    )


def _validate_inputs(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype):
    if input.ndim != 2 or mat2.ndim != 2:
        raise ValueError("Both input matrices must be two-dimensional.")

    if input.shape[1] != mat2.shape[0]:
        raise ValueError(
            "The contracting dimensions must match, but got "
            f"input.shape={tuple(input.shape)} and mat2.shape={tuple(mat2.shape)}."
        )

    if input.dtype not in _FP8_DTYPES or mat2.dtype not in _FP8_DTYPES:
        raise TypeError(
            "Both input matrices must use a supported FP8 dtype, but got "
            f"input.dtype={input.dtype} and mat2.dtype={mat2.dtype}."
        )

    if input.device != mat2.device:
        raise ValueError("Both input matrices must be on the same device.")

    if scale_a.device != input.device or scale_b.device != input.device:
        raise ValueError("The scale tensors and input matrices must share a device.")

    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise TypeError("Block scale tensors must use torch.float32.")

    if not input.is_contiguous():
        raise ValueError("input must be row-major contiguous.")

    if mat2.stride(0) != 1:
        raise ValueError("mat2 must be column-major with stride(0) equal to 1.")

    if not scale_a.is_contiguous() or not scale_b.is_contiguous():
        raise ValueError("Both block scale tensors must be contiguous.")

    if bias is not None:
        if bias.numel() != mat2.shape[1]:
            raise ValueError(
                f"bias must contain {mat2.shape[1]} elements, but got {bias.numel()}."
            )

        if bias.device != input.device:
            raise ValueError("bias and the input matrices must share a device.")

        if bias.dtype != out_dtype:
            raise TypeError(
                f"bias dtype must match out_dtype={out_dtype}, but got {bias.dtype}."
            )

    if scale_result is not None:
        if scale_result.numel() != 1 or scale_result.dtype != torch.float32:
            raise ValueError("scale_result must be a float32 scalar tensor.")

        if scale_result.device != input.device:
            raise ValueError("scale_result and the input matrices must share a device.")

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("out_dtype must be torch.bfloat16 or torch.float16.")


def scaled_mm(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    # The implementation always uses FP32 accumulation, which also satisfies
    # the numerical contract when callers request the less strict fast mode.
    del use_fast_accum

    if out_dtype is None:
        out_dtype = torch.bfloat16

    _validate_inputs(
        input,
        mat2,
        scale_a,
        scale_b,
        bias,
        scale_result,
        out_dtype,
    )

    # PyTorch only applies scale_result when producing an FP8 output.  This
    # wrapper intentionally supports BF16/FP16 outputs, so the argument is
    # validated for interface compatibility but does not change the result.

    scaling_variant = _scaling_variant(input, mat2, scale_a, scale_b)
    block_size_m, block_size_n = (
        ntops.kernels.scaled_mm.specialized_block_sizes(
            scaling_variant,
            input.shape[0],
            mat2.shape[1],
        )
    )
    output_shape = (input.shape[0], mat2.shape[1])
    output = torch.empty(
        output_shape,
        dtype=out_dtype,
        device=input.device,
    )

    kernel = _cached_make(
        ntops.kernels.scaled_mm.premake,
        scaling_variant=scaling_variant,
        input_dtype=input.dtype,
        mat2_dtype=mat2.dtype,
        scale_dtype=scale_a.dtype,
        output_dtype=out_dtype,
        bias_dtype=None if bias is None else bias.dtype,
        input_shape=tuple(input.shape),
        mat2_shape=tuple(mat2.shape),
        scale_a_shape=tuple(scale_a.shape),
        scale_b_shape=tuple(scale_b.shape),
        output_shape=output_shape,
        bias_shape=None if bias is None else output_shape,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        num_warps=ntops.kernels.scaled_mm.NUM_WARPS,
        num_stages=ntops.kernels.scaled_mm.NUM_STAGES,
    )

    # The wrapper has already validated the static tensor contract, and the
    # output is freshly allocated, so the backend can safely skip duplicate
    # runtime binding and alias checks.  Other backends use the public handle.
    launch = getattr(kernel, "_launch_prevalidated_noalias", kernel)

    if bias is None:
        launch(input, mat2, scale_a, scale_b, output)
    else:
        bias = bias.reshape(1, -1).expand(input.shape[0], -1)
        launch(input, mat2, scale_a, scale_b, bias, output)

    return output


_scaled_mm = scaled_mm


__all__ = ["_scaled_mm", "scaled_mm"]

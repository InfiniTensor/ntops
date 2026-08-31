import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


SCALE_BLOCK_SIZE = 128


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _make_fp8_matrix(shape, dtype, device):
    return torch.randn(shape, dtype=torch.float32, device=device).clamp(-2, 2).to(dtype)


def _make_col_major_fp8_matrix(shape, dtype, device):
    k, n = shape

    return _make_fp8_matrix((n, k), dtype, device).t()


def _reference_scaled_mm(input, mat2, scale_a, scale_b, out_dtype, bias=None):
    m, k = input.shape
    n = mat2.shape[1]
    num_k_blocks = _ceil_div(k, SCALE_BLOCK_SIZE)

    output = torch.zeros((m, n), dtype=torch.float32, device=input.device)

    for block in range(num_k_blocks):
        start = block * SCALE_BLOCK_SIZE
        end = min(start + SCALE_BLOCK_SIZE, k)

        partial = input[:, start:end].float() @ mat2[start:end, :].float()

        if scale_a.shape[0] == m:
            current_scale_a = scale_a[:, block]
        else:
            current_scale_a = scale_a[:, block].repeat_interleave(SCALE_BLOCK_SIZE)[:m]

        if scale_b.shape[1] == n:
            current_scale_b = scale_b[block, :]
        else:
            current_scale_b = scale_b[block, :].repeat_interleave(SCALE_BLOCK_SIZE)[:n]

        output += partial * current_scale_a[:, None] * current_scale_b[None, :]

    if bias is not None:
        output += bias.float()

    return output.to(out_dtype)


def _make_scales(m, n, k, scaling_variant, device):
    num_m_blocks = _ceil_div(m, SCALE_BLOCK_SIZE)
    num_n_blocks = _ceil_div(n, SCALE_BLOCK_SIZE)
    num_k_blocks = _ceil_div(k, SCALE_BLOCK_SIZE)

    if scaling_variant == "1x128-128x128":
        shape_a = (m, num_k_blocks)
        shape_b = (num_k_blocks, num_n_blocks)
    elif scaling_variant == "1x128-1x128":
        shape_a = (m, num_k_blocks)
        shape_b = (num_k_blocks, n)
    else:
        shape_a = (num_m_blocks, num_k_blocks)
        shape_b = (num_k_blocks, n)

    scale_a = torch.rand(shape_a, dtype=torch.float32, device=device) + 0.25
    scale_b = torch.rand(shape_b, dtype=torch.float32, device=device) + 0.25

    return scale_a, scale_b


@pytest.mark.parametrize(
    "scaling_variant, m, n, expected",
    (
        ("BLOCK_WISE_1X128_128X128", 1, 4096, (16, 128)),
        ("BLOCK_WISE_1X128_128X128", 2, 4096, (16, 128)),
        ("BLOCK_WISE_1X128_128X128", 8, 4096, (16, 128)),
        ("BLOCK_WISE_1X128_128X128", 16, 4096, (16, 128)),
        ("BLOCK_WISE_1X128_128X128", 128, 4096, (64, 128)),
        ("BLOCK_WISE_1X128_1X128", 96, 384, (32, 128)),
        ("BLOCK_WISE_128X128_1X128", 256, 96, (128, 32)),
        ("BLOCK_WISE_1X128_1X128", 33, 257, (16, 16)),
    ),
)
def test_specialized_block_sizes(scaling_variant, m, n, expected):
    variant = getattr(ntops.kernels.scaled_mm.ScalingVariant, scaling_variant)

    assert ntops.kernels.scaled_mm.specialized_block_sizes(variant, m, n) == expected


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype_name", ("float8_e4m3fnuz", "float8_e5m2fnuz"))
def test_scaled_mm_rejects_platform_incompatible_fnuz_dtypes(dtype_name):
    dtype = getattr(torch, dtype_name, None)

    if dtype is None:
        pytest.skip(f"torch.{dtype_name} is unavailable")

    device = "cuda"
    m, n, k = 16, 128, 128
    input = _make_fp8_matrix((m, k), dtype, device)
    mat2 = _make_col_major_fp8_matrix((k, n), dtype, device)
    scale_a, scale_b = _make_scales(m, n, k, "1x128-128x128", device)

    with pytest.raises(TypeError, match="supported FP8 dtype"):
        ntops.torch.scaled_mm(input, mat2, scale_a, scale_b)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "m, n, k",
    (
        (1, 128, 128),
        (16, 128, 128),
        (33, 257, 385),
    ),
)
@pytest.mark.parametrize(
    "scaling_variant",
    (
        "1x128-128x128",
        "1x128-1x128",
        "128x128-1x128",
    ),
)
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
def test_scaled_mm(m, n, k, scaling_variant, out_dtype):
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)

    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")

    device = "cuda"
    input = _make_fp8_matrix((m, k), fp8_dtype, device)
    mat2 = _make_col_major_fp8_matrix((k, n), fp8_dtype, device)
    scale_a, scale_b = _make_scales(m, n, k, scaling_variant, device)

    actual = ntops.torch.scaled_mm(
        input,
        mat2,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )
    expected = _reference_scaled_mm(input, mat2, scale_a, scale_b, out_dtype)

    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


@skip_if_cuda_not_available
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
def test_scaled_mm_bias_and_ignored_scale_result_for_high_precision_output(
    out_dtype,
):
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)

    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")

    device = "cuda"
    m, n, k = 16, 128, 128
    input = _make_fp8_matrix((m, k), fp8_dtype, device)
    mat2 = _make_col_major_fp8_matrix((k, n), fp8_dtype, device)
    scale_a, scale_b = _make_scales(m, n, k, "1x128-128x128", device)
    bias = torch.randn(n, dtype=out_dtype, device=device)
    # PyTorch applies scale_result only when the requested output is FP8.
    scale_result = torch.tensor(0.25, dtype=torch.float32, device=device)

    actual = ntops.torch.scaled_mm(
        input,
        mat2,
        scale_a,
        scale_b,
        bias=bias,
        scale_result=scale_result,
        out_dtype=out_dtype,
    )
    expected = _reference_scaled_mm(
        input,
        mat2,
        scale_a,
        scale_b,
        out_dtype,
        bias,
    )

    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

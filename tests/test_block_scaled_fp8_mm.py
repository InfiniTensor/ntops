import importlib
import inspect
import pathlib

import ninetoothed
import pytest
import torch
import torch.nn.functional as F

import ntops

skip_if_cuda_is_unavailable = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-compatible accelerator is unavailable",
)


def _column_major(value):
    return value.t().contiguous().t()


def _make_inputs(m, n, k, recipe_b, output_dtype=torch.bfloat16, bias=False):
    device = "cuda"
    k_blocks = k // 128
    n_blocks = n // 128
    mat_a = torch.randn((m, k), device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    weight = torch.randn((n, k), device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    mat_b = weight.t()
    scale_a_logical = torch.rand((m, k_blocks), device=device) + 0.25
    scale_a = _column_major(scale_a_logical)

    if recipe_b == ntops.torch.ScalingType.BlockWise1x128:
        scale_b_logical = torch.rand((n, k_blocks), device=device) + 0.25
        scale_b = _column_major(scale_b_logical)
        weight_dequantized = (
            weight.float().reshape(n, k_blocks, 128) * scale_b_logical[..., None]
        ).reshape(n, k)
    else:
        scale_b_logical = torch.rand((n_blocks, k_blocks), device=device) + 0.25
        padded_k_blocks = ((k_blocks + 3) // 4) * 4
        scale_b_padded = F.pad(scale_b_logical, (0, padded_k_blocks - k_blocks))
        scale_b = scale_b_padded.t()
        weight_dequantized = (
            weight.float().reshape(n_blocks, 128, k_blocks, 128)
            * scale_b_logical[:, None, :, None]
        ).reshape(n, k)

    mat_a_dequantized = (
        mat_a.float().reshape(m, k_blocks, 128) * scale_a_logical[..., None]
    ).reshape(m, k)
    bias_value = None
    if bias:
        bias_value = torch.randn((n,), device=device, dtype=output_dtype)
    expected = mat_a_dequantized @ weight_dequantized.t()
    if bias_value is not None:
        expected = expected + bias_value.float()
    return mat_a, mat_b, scale_a, scale_b, bias_value, expected.to(output_dtype)


def _block_scaled_fp8_mm(mat_a, mat_b, scale_a, scale_b, recipe_b, **kwargs):
    return ntops.torch.block_scaled_fp8_mm(
        mat_a,
        mat_b,
        scale_a,
        ntops.torch.ScalingType.BlockWise1x128,
        scale_b,
        recipe_b,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("hip_version", "is_corex", "cuda_version", "capability", "expected"),
    (
        ("6.3", False, None, None, (32, 1)),
        (None, True, "10.2", (9, 0), (16, 4)),
        (None, False, "12.4", (8, 9), (32, 4)),
        (None, False, "12.4", (9, 0), (32, 4)),
        (None, False, "12.4", (8, 6), (16, 4)),
        (None, False, None, None, (16, 4)),
    ),
)
def test_block_scaled_fp8_mm_selects_dot_path_by_backend_capability(
    monkeypatch,
    hip_version,
    is_corex,
    cuda_version,
    capability,
    expected,
):
    module = importlib.import_module("ntops.torch.block_scaled_fp8_mm")
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch, "corex", is_corex, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_version is not None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability,
    )

    assert module._kernel_launch_config(torch.device("cuda")) == expected


def test_block_scaled_fp8_mm_keeps_fixed_hip_config(monkeypatch):
    module = importlib.import_module("ntops.torch.block_scaled_fp8_mm")
    monkeypatch.setattr(torch.version, "hip", "6.3")

    assert module._kernel_tuning_config(torch.device("cuda")) == (32, 1, 1, 1)


def test_block_scaled_fp8_mm_keeps_fixed_non_hip_config(monkeypatch):
    module = importlib.import_module("ntops.torch.block_scaled_fp8_mm")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "corex", True, raising=False)

    assert module._kernel_tuning_config(torch.device("cuda")) == (16, 4, 1, 1)


@skip_if_cuda_is_unavailable
@pytest.mark.parametrize(
    "recipe_b",
    (
        ntops.torch.ScalingType.BlockWise1x128,
        ntops.torch.ScalingType.BlockWise128x128,
    ),
)
def test_block_scaled_fp8_mm_block_fp8_numerics(recipe_b):
    torch.manual_seed(0)
    has_bias = recipe_b == ntops.torch.ScalingType.BlockWise128x128
    inputs = _make_inputs(17, 256, 256, recipe_b, bias=has_bias)
    mat_a, mat_b, scale_a, scale_b, bias, expected = inputs
    output = _block_scaled_fp8_mm(mat_a, mat_b, scale_a, scale_b, recipe_b, bias=bias)

    assert output.shape == expected.shape
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.125)


@skip_if_cuda_is_unavailable
def test_block_scaled_fp8_mm_attention_decode_with_bias_and_fp32_output():
    torch.manual_seed(1)
    recipe_b = ntops.torch.ScalingType.BlockWise128x128
    inputs = _make_inputs(1, 128, 128, recipe_b, torch.float32, bias=True)
    mat_a, mat_b, scale_a, scale_b, bias, expected = inputs
    output = _block_scaled_fp8_mm(
        mat_a,
        mat_b,
        [scale_a],
        [scale_b],
        recipe_b,
        bias=bias,
        output_dtype=torch.float32,
        swizzle_a=[ntops.torch.SwizzleType.NO_SWIZZLE],
        swizzle_b=ntops.torch.SwizzleType.NO_SWIZZLE,
    )

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.1)


@skip_if_cuda_is_unavailable
def test_block_scaled_fp8_mm_covers_the_full_128_element_scale_block():
    recipe_a = ntops.torch.ScalingType.BlockWise1x128
    recipe_b = ntops.torch.ScalingType.BlockWise128x128
    mat_a = torch.zeros((1, 4, 32), device="cuda")
    mat_a[:, :, 16:] = 1
    mat_a = mat_a.reshape(1, 128).to(torch.float8_e4m3fn)
    column_values = torch.arange(128, device="cuda") % 7 - 3
    weight = column_values[:, None].expand(-1, 128)
    mat_b = weight.to(torch.float8_e4m3fn).t()
    scale_a = torch.ones((1, 1), device="cuda")
    scale_b = torch.ones((1, 1), device="cuda")

    output = ntops.torch.block_scaled_fp8_mm(
        mat_a,
        mat_b,
        scale_a,
        recipe_a,
        scale_b,
        recipe_b,
        output_dtype=torch.float32,
    )

    expected = (column_values * 64).to(torch.float32).unsqueeze(0)
    torch.testing.assert_close(output, expected)


def _cpu_inputs(m=2, n=128, k=128):
    mat_a = torch.empty((m, k), dtype=torch.float8_e4m3fn)
    mat_b = torch.empty((n, k), dtype=torch.float8_e4m3fn).t()
    scale_a = _column_major(torch.ones((m, k // 128)))
    scale_b = torch.ones((k // 128, n // 128))
    return mat_a, mat_b, scale_a, scale_b


def test_block_scaled_fp8_mm_public_name_and_signature():
    expected = (
        "mat_a",
        "mat_b",
        "scale_a",
        "scale_recipe_a",
        "scale_b",
        "scale_recipe_b",
        "swizzle_a",
        "swizzle_b",
        "bias",
        "output_dtype",
        "contraction_dim",
        "use_fast_accum",
    )
    assert not hasattr(ntops.torch, "scaled_mm")
    assert (
        tuple(inspect.signature(ntops.torch.block_scaled_fp8_mm).parameters) == expected
    )


def test_block_scaled_fp8_mm_accepts_list_api_and_empty_m():
    mat_a, mat_b, scale_a, scale_b = _cpu_inputs(m=0)
    output = ntops.torch.block_scaled_fp8_mm(
        mat_a,
        mat_b,
        [scale_a],
        [ntops.torch.ScalingType.BlockWise1x128],
        [scale_b],
        [ntops.torch.ScalingType.BlockWise128x128],
        swizzle_a=[ntops.torch.SwizzleType.NO_SWIZZLE],
        swizzle_b=[ntops.torch.SwizzleType.NO_SWIZZLE],
        output_dtype=None,
        contraction_dim=[],
    )
    assert output.shape == (0, mat_b.shape[1])
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    (
        (
            "scale_recipe_a",
            ntops.torch.ScalingType.TensorWise,
            NotImplementedError,
        ),
        (
            "scale_recipe_b",
            ntops.torch.ScalingType.RowWise,
            NotImplementedError,
        ),
        (
            "swizzle_a",
            ntops.torch.SwizzleType.SWIZZLE_32_4_4,
            NotImplementedError,
        ),
        (
            "scale_recipe_b",
            [
                ntops.torch.ScalingType.BlockWise128x128,
                ntops.torch.ScalingType.BlockWise128x128,
            ],
            NotImplementedError,
        ),
        ("output_dtype", torch.int32, ValueError),
        ("contraction_dim", (1,), NotImplementedError),
        ("use_fast_accum", True, NotImplementedError),
    ),
)
def test_block_scaled_fp8_mm_rejects_unsupported_options(argument, value, error):
    mat_a, mat_b, scale_a, scale_b = _cpu_inputs()
    arguments = {
        "scale_recipe_a": ntops.torch.ScalingType.BlockWise1x128,
        "scale_recipe_b": ntops.torch.ScalingType.BlockWise128x128,
    }
    arguments[argument] = value
    with pytest.raises(error):
        ntops.torch.block_scaled_fp8_mm(
            mat_a, mat_b, scale_a, scale_b=scale_b, **arguments
        )


def test_block_scaled_fp8_mm_validates_shapes_dtypes_and_layouts():
    mat_a, mat_b, scale_a, scale_b = _cpu_inputs()
    recipe_b = ntops.torch.ScalingType.BlockWise128x128

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        _block_scaled_fp8_mm(mat_a.float(), mat_b, scale_a, scale_b, recipe_b)
    with pytest.raises(ValueError, match="scale_a must have shape"):
        _block_scaled_fp8_mm(mat_a, mat_b, scale_a[:, :0], scale_b, recipe_b)
    with pytest.raises(ValueError, match="column-major"):
        _block_scaled_fp8_mm(mat_a, mat_b.contiguous(), scale_a, scale_b, recipe_b)
    bad_n = torch.empty((129, 128), dtype=torch.float8_e4m3fn).t()
    with pytest.raises(ValueError, match="N must be a multiple"):
        _block_scaled_fp8_mm(mat_a, bad_n, scale_a, scale_b, recipe_b)


def _generated_kernel_source():
    kernel = ninetoothed.make(
        *ntops.kernels.block_scaled_fp8_mm.premake(torch.float8_e4m3fn, torch.bfloat16),
        num_warps=1,
        num_stages=1,
        max_num_configs=1,
    )
    if hasattr(kernel, "_compilation"):
        sources = (
            value
            for name, value in kernel._compilation.artifact.sources.items()
            if name.endswith(".py")
        )
        return "\n".join(str(value) for value in sources)
    return pathlib.Path(kernel._source).read_text()


def test_block_scaled_fp8_mm_lowers_to_portable_fp8_dot():
    source = _generated_kernel_source()
    dot_count = source.count("tl.dot(") + source.count("triton.language.dot(")
    assert dot_count == 1
    assert (
        "def block_scaled_fp8_mm_k32(" in source
        or "def block_scaled_fp8_mm_k32_kernel(" in source
    )
    assert "dot_scaled" not in source
    assert "ntl." not in source

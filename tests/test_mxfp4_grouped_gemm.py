"""Correctness tests for the PyTorch-compatible MXFP4 grouped GEMM."""

import importlib

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _decode_e2m1(packed):
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)

    return table[packed.to(torch.long)]


def _canonical_offsets(offsets, num_experts):
    if len(offsets) == num_experts + 1:
        return offsets

    return (0, *offsets)


def _reference(input, packed_weight, weight_scale, offsets):
    boundaries = _canonical_offsets(offsets, packed_weight.shape[0])
    outputs = []

    for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        packed = packed_weight[expert]
        unpacked = torch.empty(
            packed.shape[:-1] + (packed.shape[-1] * 2,),
            dtype=torch.float32,
            device=packed.device,
        )
        unpacked[..., 0::2] = _decode_e2m1(packed & 0x0F)
        unpacked[..., 1::2] = _decode_e2m1(packed >> 4)
        scales = torch.pow(
            2.0,
            weight_scale[expert].to(torch.float32) - 127.0,
        )
        unpacked *= scales.repeat_interleave(32, dim=-1)
        outputs.append(input[start:end].float() @ unpacked.transpose(0, 1))

    return torch.cat(outputs, dim=0).to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_mxfp4_grouped_gemm_ragged(dtype):
    torch.manual_seed(7)
    experts, n, k = 3, 32, 64
    offsets = torch.tensor((1, 4, 6), device="cuda", dtype=torch.int32)
    input = torch.randn(6, k, device="cuda", dtype=dtype)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = (
        torch.tensor(
            [[[126, 127]], [[127, 128]], [[125, 126]]],
            device="cuda",
            dtype=torch.uint8,
        )
        .expand(experts, n, k // 32)
        .contiguous()
    )

    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@skip_if_cuda_not_available
def test_mxfp4_grouped_gemm_accepts_legacy_boundaries():
    torch.manual_seed(11)
    experts, n, k = 2, 16, 32
    input = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.full(
        (experts, n, k // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    offsets = torch.tensor((0, 1, 3), device="cuda", dtype=torch.int32)

    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@skip_if_cuda_not_available
def test_mxfp4_grouped_gemm_zero_token_expert():
    torch.manual_seed(13)
    experts, n, k = 3, 32, 64
    input = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.randint(
        124,
        130,
        (experts, n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    offsets = torch.tensor((0, 3, 3), device="cuda", dtype=torch.int32)

    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "experts,n,k,counts",
    (
        (1, 16, 32, (1,)),
        (2, 32, 64, (1, 3)),
        (2, 64, 128, (5, 2)),
        (4, 128, 256, (1, 0, 17, 5)),
    ),
)
def test_mxfp4_grouped_gemm_shape_matrix(experts, n, k, counts):
    torch.manual_seed(19 + n + k)
    offsets = torch.tensor(
        torch.tensor(counts).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    input = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.randint(
        124,
        130,
        (experts, n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )

    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_mxfp4_encoding_contract():
    assert ntops.kernels.mxfp4_grouped_gemm.MXFP4_BLOCK_SIZE == 32
    assert ntops.kernels.mxfp4_grouped_gemm.PACKED_VALUES_PER_BYTE == 2
    assert ntops.kernels.mxfp4_grouped_gemm.PACKED_VALUES_PER_SCALE == 16
    assert _E2M1_VALUES == (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )


def test_mxfp4_column_tile_uses_shape_derived_long_reduction_path():
    wrapper = importlib.import_module("ntops.torch.mxfp4_grouped_gemm")

    assert wrapper._select_column_tile(256, 1024, 7) == 64
    assert wrapper._select_column_tile(256, 512, 7) == 128
    assert wrapper._select_column_tile(256, 1024, 32) == 128


@skip_if_cuda_not_available
def test_offsets_metadata_reuses_padding_and_tracks_actual_max():
    wrapper = importlib.import_module("ntops.torch.mxfp4_grouped_gemm")
    offsets = torch.tensor((1, 4, 6), device="cuda", dtype=torch.int32)

    first = wrapper._offsets_metadata(offsets, 3, 6)
    second = wrapper._offsets_metadata(offsets, 3, 6)

    assert first is second
    assert first.padded_offsets is second.padded_offsets
    assert first.padded_offsets.tolist() == [0, 1, 4, 6]
    assert first.max_tokens_per_expert == 3

    offsets[1] = 5
    updated = wrapper._offsets_metadata(offsets, 3, 6)

    assert updated is not first
    assert updated.padded_offsets.tolist() == [0, 1, 5, 6]
    assert updated.max_tokens_per_expert == 4


@skip_if_cuda_not_available
def test_offsets_metadata_validates_routing_contract():
    wrapper = importlib.import_module("ntops.torch.mxfp4_grouped_gemm")
    decreasing = torch.tensor((2, 1), device="cuda", dtype=torch.int32)
    incomplete = torch.tensor((1, 3), device="cuda", dtype=torch.int32)
    missing_zero = torch.tensor((1, 2, 4), device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="monotonically nondecreasing"):
        wrapper._offsets_metadata(decreasing, 2, 1)

    with pytest.raises(ValueError, match="input.shape"):
        wrapper._offsets_metadata(incomplete, 2, 4)

    with pytest.raises(ValueError, match="start at zero"):
        wrapper._offsets_metadata(missing_zero, 2, 4)


@skip_if_cuda_not_available
def test_mxfp4_grouped_gemm_reuses_bound_launch_with_dynamic_data():
    torch.manual_seed(23)
    experts, n, k = 3, 32, 64
    offsets = torch.tensor((1, 4, 6), device="cuda", dtype=torch.int32)
    input = torch.randn(6, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.randint(
        124,
        130,
        (experts, n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    output = torch.empty(6, n, device="cuda", dtype=torch.bfloat16)

    ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
        out=output,
    )
    input.add_(0.25)
    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
        out=output,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    input.add_(0.125)
    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
        out=output,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    with pytest.raises(ValueError, match="num_experts"):
        ntops.torch.mxfp4_grouped_gemm(
            input,
            packed_weight,
            weight_scale,
            offsets,
            num_experts=experts - 1,
            out=output,
        )

    offsets[0] = 2
    actual = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
        out=output,
    )
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@skip_if_cuda_not_available
def test_mxfp4_grouped_gemm_bound_launch_uses_current_stream():
    torch.manual_seed(31)
    experts, n, k = 2, 32, 64
    offsets = torch.tensor((2, 4), device="cuda", dtype=torch.int32)
    input = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.randint(
        124,
        130,
        (experts, n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    output = torch.empty(4, n, device="cuda", dtype=torch.bfloat16)

    ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
        out=output,
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        input.add_(0.25)
        actual = ntops.torch.mxfp4_grouped_gemm(
            input,
            packed_weight,
            weight_scale,
            offsets,
            out=output,
        )

    stream.synchronize()
    expected = _reference(input, packed_weight, weight_scale, offsets.tolist())

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@skip_if_cuda_not_available
def test_mxfp4_grouped_gemm_new_outputs_remain_independent():
    torch.manual_seed(29)
    experts, n, k = 2, 32, 64
    offsets = torch.tensor((2, 4), device="cuda", dtype=torch.int32)
    input = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
    )
    weight_scale = torch.randint(
        124,
        130,
        (experts, n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    first = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected_first = _reference(
        input,
        packed_weight,
        weight_scale,
        offsets.tolist(),
    )
    first_snapshot = first.clone()

    input.add_(0.25)
    second = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected_second = _reference(
        input,
        packed_weight,
        weight_scale,
        offsets.tolist(),
    )
    input.add_(0.125)
    third = ntops.torch.mxfp4_grouped_gemm(
        input,
        packed_weight,
        weight_scale,
        offsets,
    )
    expected_third = _reference(
        input,
        packed_weight,
        weight_scale,
        offsets.tolist(),
    )

    assert first.data_ptr() != second.data_ptr()
    assert second.data_ptr() != third.data_ptr()
    torch.testing.assert_close(first, first_snapshot)
    torch.testing.assert_close(first, expected_first, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(second, expected_second, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(third, expected_third, rtol=3e-2, atol=3e-2)

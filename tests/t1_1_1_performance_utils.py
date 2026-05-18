from dataclasses import dataclass

import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_MIN_TORCH_SPEED_RATIO = 0.9
_LARGE_NUMEL = 1 << 24
_MID_NUMEL = _LARGE_NUMEL
_SMALL_NUMEL = _LARGE_NUMEL


@dataclass(frozen=True)
class PerfCase:
    op_name: str
    case_name: str
    make_pair: object


def _rand_float(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cuda")


def _rand_lgamma(shape, dtype):
    return torch.rand(shape, dtype=dtype, device="cuda") * 8 + 0.25


def _rand_int(shape, dtype, low=-128, high=128):
    return torch.randint(low, high, shape, dtype=dtype, device="cuda")


def _noncontig_float(side, dtype):
    return torch.randn((side, side), dtype=dtype, device="cuda").t()


def _permute3d_float(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cuda").permute(2, 0, 1)


def _noncontig_lgamma(side, dtype):
    return (torch.rand((side, side), dtype=dtype, device="cuda") * 8 + 0.25).t()


def _permute3d_lgamma(shape, dtype):
    return (torch.rand(shape, dtype=dtype, device="cuda") * 8 + 0.25).permute(2, 0, 1)


def _noncontig_int(side, dtype, low=-128, high=128):
    return torch.randint(low, high, (side, side), dtype=dtype, device="cuda").t()


def _permute3d_int(shape, dtype, low=-128, high=128):
    return torch.randint(low, high, shape, dtype=dtype, device="cuda").permute(2, 0, 1)


def _make_unary(op, ref, factory, shape, dtype, out=False):
    def make_pair():
        input = factory(shape, dtype)
        if not out:
            return lambda: op(input), lambda: ref(input)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, out=nt_out), lambda: ref(input, out=th_out)

    return make_pair


def _make_unary_noncontig(op, ref, factory, side, dtype, out=False):
    def make_pair():
        input = factory(side, dtype)
        if not out:
            return lambda: op(input), lambda: ref(input)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, out=nt_out), lambda: ref(input, out=th_out)

    return make_pair


def _make_unary_permute3d(op, ref, factory, shape, dtype, out=False):
    def make_pair():
        input = factory(shape, dtype)
        if not out:
            return lambda: op(input), lambda: ref(input)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, out=nt_out), lambda: ref(input, out=th_out)

    return make_pair


def _make_binary(op, ref, shape, dtype, out=False):
    def make_pair():
        input = _rand_float(shape, dtype)
        other = _rand_float(shape, dtype)
        if not out:
            return lambda: op(input, other), lambda: ref(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, other, out=nt_out), lambda: ref(input, other, out=th_out)

    return make_pair


def _make_binary_noncontig(op, ref, side, dtype, out=False):
    def make_pair():
        input = _noncontig_float(side, dtype)
        other = _noncontig_float(side, dtype)
        if not out:
            return lambda: op(input, other), lambda: ref(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, other, out=nt_out), lambda: ref(input, other, out=th_out)

    return make_pair


def _make_binary_permute3d(op, ref, shape, dtype, out=False):
    def make_pair():
        input = _permute3d_float(shape, dtype)
        other = _permute3d_float(shape, dtype)
        if not out:
            return lambda: op(input, other), lambda: ref(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: op(input, other, out=nt_out), lambda: ref(input, other, out=th_out)

    return make_pair


def _make_binary_broadcast(op, ref, side, dtype):
    def make_pair():
        input = _rand_float((side, 1), dtype)
        other = _rand_float((1, side), dtype)
        return lambda: op(input, other), lambda: ref(input, other)

    return make_pair


def _make_binary_broadcast_rect(op, ref, rows, cols, dtype):
    def make_pair():
        input = _rand_float((rows, 1), dtype)
        other = _rand_float((1, cols), dtype)
        return lambda: op(input, other), lambda: ref(input, other)

    return make_pair


def _make_lcm(shape, dtype, low=-128, high=128, out=False):
    def make_pair():
        input = _rand_int(shape, dtype, low, high)
        other = _rand_int(shape, dtype, low, high)
        if not out:
            return lambda: ntops.torch.lcm(input, other), lambda: torch.lcm(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: ntops.torch.lcm(input, other, out=nt_out), lambda: torch.lcm(input, other, out=th_out)

    return make_pair


def _make_lcm_noncontig(side, dtype, low=-128, high=128, out=False):
    def make_pair():
        input = _noncontig_int(side, dtype, low, high)
        other = _noncontig_int(side, dtype, low, high)
        if not out:
            return lambda: ntops.torch.lcm(input, other), lambda: torch.lcm(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: ntops.torch.lcm(input, other, out=nt_out), lambda: torch.lcm(input, other, out=th_out)

    return make_pair


def _make_lcm_permute3d(shape, dtype, low=-128, high=128, out=False):
    def make_pair():
        input = _permute3d_int(shape, dtype, low, high)
        other = _permute3d_int(shape, dtype, low, high)
        if not out:
            return lambda: ntops.torch.lcm(input, other), lambda: torch.lcm(input, other)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: ntops.torch.lcm(input, other, out=nt_out), lambda: torch.lcm(input, other, out=th_out)

    return make_pair


def _make_lcm_broadcast(side, dtype, low=-128, high=128):
    def make_pair():
        input = _rand_int((side, 1), dtype, low, high)
        other = _rand_int((1, side), dtype, low, high)
        return lambda: ntops.torch.lcm(input, other), lambda: torch.lcm(input, other)

    return make_pair


_PERF_CASES = [
    PerfCase("rad2deg", "f16_large_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_LARGE_NUMEL,), torch.float16)),
    PerfCase("rad2deg", "f32_large_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_LARGE_NUMEL,), torch.float32)),
    PerfCase("rad2deg", "f64_large_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_LARGE_NUMEL,), torch.float64)),
    PerfCase("rad2deg", "f32_large_2d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (4096, 4096), torch.float32)),
    PerfCase("rad2deg", "f16_large_3d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (256, 256, 256), torch.float16)),
    PerfCase("rad2deg", "f32_large_3d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (256, 256, 256), torch.float32)),
    PerfCase("rad2deg", "f64_large_3d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (256, 256, 256), torch.float64)),
    PerfCase("rad2deg", "f32_large_out_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_LARGE_NUMEL,), torch.float32, out=True)),
    PerfCase("rad2deg", "f64_large_out_2d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (4096, 4096), torch.float64, out=True)),
    PerfCase("rad2deg", "f16_large_out_3d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (256, 256, 256), torch.float16, out=True)),
    PerfCase("rad2deg", "f32_mid_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_MID_NUMEL,), torch.float32)),
    PerfCase("rad2deg", "f16_mid_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_MID_NUMEL,), torch.float16)),
    PerfCase("rad2deg", "f64_mid_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_MID_NUMEL,), torch.float64)),
    PerfCase("rad2deg", "f32_small_1d", _make_unary(ntops.torch.rad2deg, torch.rad2deg, _rand_float, (_SMALL_NUMEL,), torch.float32)),
    PerfCase("rad2deg", "f16_noncontig_4096", _make_unary_noncontig(ntops.torch.rad2deg, torch.rad2deg, _noncontig_float, 4096, torch.float16)),
    PerfCase("rad2deg", "f32_noncontig_4096", _make_unary_noncontig(ntops.torch.rad2deg, torch.rad2deg, _noncontig_float, 4096, torch.float32)),
    PerfCase("rad2deg", "f64_noncontig_2048", _make_unary_noncontig(ntops.torch.rad2deg, torch.rad2deg, _noncontig_float, 2048, torch.float64)),
    PerfCase("rad2deg", "f32_noncontig_out_4096", _make_unary_noncontig(ntops.torch.rad2deg, torch.rad2deg, _noncontig_float, 4096, torch.float32, out=True)),
    PerfCase("rad2deg", "f32_permute3d_256x256x128", _make_unary_permute3d(ntops.torch.rad2deg, torch.rad2deg, _permute3d_float, (256, 256, 128), torch.float32)),
    PerfCase("rad2deg", "f32_permute3d_out_256x256x128", _make_unary_permute3d(ntops.torch.rad2deg, torch.rad2deg, _permute3d_float, (256, 256, 128), torch.float32, out=True)),
    PerfCase("copysign", "f16_large_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_LARGE_NUMEL,), torch.float16)),
    PerfCase("copysign", "f32_large_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_LARGE_NUMEL,), torch.float32)),
    PerfCase("copysign", "f64_large_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_LARGE_NUMEL,), torch.float64)),
    PerfCase("copysign", "f32_large_2d", _make_binary(ntops.torch.copysign, torch.copysign, (4096, 4096), torch.float32)),
    PerfCase("copysign", "f16_large_3d", _make_binary(ntops.torch.copysign, torch.copysign, (256, 256, 256), torch.float16)),
    PerfCase("copysign", "f32_large_3d", _make_binary(ntops.torch.copysign, torch.copysign, (256, 256, 256), torch.float32)),
    PerfCase("copysign", "f64_large_3d", _make_binary(ntops.torch.copysign, torch.copysign, (256, 256, 256), torch.float64)),
    PerfCase("copysign", "f32_large_out_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_LARGE_NUMEL,), torch.float32, out=True)),
    PerfCase("copysign", "f64_large_out_2d", _make_binary(ntops.torch.copysign, torch.copysign, (4096, 4096), torch.float64, out=True)),
    PerfCase("copysign", "f16_large_out_3d", _make_binary(ntops.torch.copysign, torch.copysign, (256, 256, 256), torch.float16, out=True)),
    PerfCase("copysign", "f32_mid_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_MID_NUMEL,), torch.float32)),
    PerfCase("copysign", "f16_mid_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_MID_NUMEL,), torch.float16)),
    PerfCase("copysign", "f64_mid_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_MID_NUMEL,), torch.float64)),
    PerfCase("copysign", "f32_small_1d", _make_binary(ntops.torch.copysign, torch.copysign, (_SMALL_NUMEL,), torch.float32)),
    PerfCase("copysign", "f32_broadcast_rect_2048x8192", _make_binary_broadcast_rect(ntops.torch.copysign, torch.copysign, 2048, 8192, torch.float32)),
    PerfCase("copysign", "f32_broadcast_4096", _make_binary_broadcast(ntops.torch.copysign, torch.copysign, 4096, torch.float32)),
    PerfCase("copysign", "f16_noncontig_4096", _make_binary_noncontig(ntops.torch.copysign, torch.copysign, 4096, torch.float16)),
    PerfCase("copysign", "f32_noncontig_4096", _make_binary_noncontig(ntops.torch.copysign, torch.copysign, 4096, torch.float32)),
    PerfCase("copysign", "f64_noncontig_2048", _make_binary_noncontig(ntops.torch.copysign, torch.copysign, 2048, torch.float64)),
    PerfCase("copysign", "f32_permute3d_out_256x256x128", _make_binary_permute3d(ntops.torch.copysign, torch.copysign, (256, 256, 128), torch.float32, out=True)),
    PerfCase("lcm", "i32_large_1d", _make_lcm((_LARGE_NUMEL,), torch.int32, -128, 128)),
    PerfCase("lcm", "i32_large_positive_1d", _make_lcm((_LARGE_NUMEL,), torch.int32, 1, 32)),
    PerfCase("lcm", "i32_large_2d", _make_lcm((4096, 4096), torch.int32, -128, 128)),
    PerfCase("lcm", "i32_large_positive_2d", _make_lcm((4096, 4096), torch.int32, 1, 32)),
    PerfCase("lcm", "i32_large_3d", _make_lcm((256, 256, 256), torch.int32, -128, 128)),
    PerfCase("lcm", "i32_large_positive_3d", _make_lcm((256, 256, 256), torch.int32, 1, 32)),
    PerfCase("lcm", "i32_large_out_1d", _make_lcm((_LARGE_NUMEL,), torch.int32, -128, 128, out=True)),
    PerfCase("lcm", "i32_large_out_2d", _make_lcm((4096, 4096), torch.int32, -128, 128, out=True)),
    PerfCase("lcm", "i32_broadcast_8192", _make_lcm_broadcast(8192, torch.int32, -128, 128)),
    PerfCase("lcm", "i32_large_low_1d", _make_lcm((_LARGE_NUMEL,), torch.int32, -8, 9)),
    PerfCase("lcm", "i16_mid_1d", _make_lcm((_MID_NUMEL,), torch.int16, -128, 128)),
    PerfCase("lcm", "i16_large_1d", _make_lcm((_LARGE_NUMEL,), torch.int16, -128, 128)),
    PerfCase("lcm", "i64_mid_1d", _make_lcm((_MID_NUMEL,), torch.int64, -128, 128)),
    PerfCase("lcm", "i64_large_1d", _make_lcm((_LARGE_NUMEL,), torch.int64, -128, 128)),
    PerfCase("lcm", "u8_mid_1d", _make_lcm((_MID_NUMEL,), torch.uint8, 1, 128)),
    PerfCase("lcm", "i8_mid_1d", _make_lcm((_MID_NUMEL,), torch.int8, -64, 64)),
    PerfCase("lcm", "i32_noncontig_4096", _make_lcm_noncontig(4096, torch.int32, -128, 128)),
    PerfCase("lcm", "i32_noncontig_out_4096", _make_lcm_noncontig(4096, torch.int32, -128, 128, out=True)),
    PerfCase("lcm", "i16_noncontig_6144", _make_lcm_noncontig(6144, torch.int16, -128, 128)),
    PerfCase("lcm", "i32_permute3d_out_256x256x128", _make_lcm_permute3d((256, 256, 128), torch.int32, -128, 128, out=True)),
    PerfCase("lgamma", "f16_large_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_LARGE_NUMEL,), torch.float16)),
    PerfCase("lgamma", "f32_large_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_LARGE_NUMEL,), torch.float32)),
    PerfCase("lgamma", "f64_large_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_LARGE_NUMEL,), torch.float64)),
    PerfCase("lgamma", "f32_large_2d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (4096, 4096), torch.float32)),
    PerfCase("lgamma", "f16_large_3d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (256, 256, 256), torch.float16)),
    PerfCase("lgamma", "f32_large_3d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (256, 256, 256), torch.float32)),
    PerfCase("lgamma", "f64_large_3d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (256, 256, 256), torch.float64)),
    PerfCase("lgamma", "f32_large_out_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_LARGE_NUMEL,), torch.float32, out=True)),
    PerfCase("lgamma", "f64_large_out_2d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (4096, 4096), torch.float64, out=True)),
    PerfCase("lgamma", "f16_large_out_3d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (256, 256, 256), torch.float16, out=True)),
    PerfCase("lgamma", "f32_mid_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_MID_NUMEL,), torch.float32)),
    PerfCase("lgamma", "f16_mid_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_MID_NUMEL,), torch.float16)),
    PerfCase("lgamma", "f64_mid_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_MID_NUMEL,), torch.float64)),
    PerfCase("lgamma", "f32_small_1d", _make_unary(ntops.torch.lgamma, torch.lgamma, _rand_lgamma, (_SMALL_NUMEL,), torch.float32)),
    PerfCase("lgamma", "f16_noncontig_4096", _make_unary_noncontig(ntops.torch.lgamma, torch.lgamma, _noncontig_lgamma, 4096, torch.float16)),
    PerfCase("lgamma", "f32_noncontig_4096", _make_unary_noncontig(ntops.torch.lgamma, torch.lgamma, _noncontig_lgamma, 4096, torch.float32)),
    PerfCase("lgamma", "f64_noncontig_2048", _make_unary_noncontig(ntops.torch.lgamma, torch.lgamma, _noncontig_lgamma, 2048, torch.float64)),
    PerfCase("lgamma", "f32_noncontig_out_4096", _make_unary_noncontig(ntops.torch.lgamma, torch.lgamma, _noncontig_lgamma, 4096, torch.float32, out=True)),
    PerfCase("lgamma", "f32_permute3d_256x256x128", _make_unary_permute3d(ntops.torch.lgamma, torch.lgamma, _permute3d_lgamma, (256, 256, 128), torch.float32)),
    PerfCase("lgamma", "f32_permute3d_out_256x256x128", _make_unary_permute3d(ntops.torch.lgamma, torch.lgamma, _permute3d_lgamma, (256, 256, 128), torch.float32, out=True)),
    PerfCase("nextafter", "f16_large_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_LARGE_NUMEL,), torch.float16)),
    PerfCase("nextafter", "f32_large_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_LARGE_NUMEL,), torch.float32)),
    PerfCase("nextafter", "f64_large_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_LARGE_NUMEL,), torch.float64)),
    PerfCase("nextafter", "f32_large_2d", _make_binary(ntops.torch.nextafter, torch.nextafter, (4096, 4096), torch.float32)),
    PerfCase("nextafter", "f16_large_3d", _make_binary(ntops.torch.nextafter, torch.nextafter, (256, 256, 256), torch.float16)),
    PerfCase("nextafter", "f32_large_3d", _make_binary(ntops.torch.nextafter, torch.nextafter, (256, 256, 256), torch.float32)),
    PerfCase("nextafter", "f64_large_3d", _make_binary(ntops.torch.nextafter, torch.nextafter, (256, 256, 256), torch.float64)),
    PerfCase("nextafter", "f32_large_out_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_LARGE_NUMEL,), torch.float32, out=True)),
    PerfCase("nextafter", "f64_large_out_2d", _make_binary(ntops.torch.nextafter, torch.nextafter, (4096, 4096), torch.float64, out=True)),
    PerfCase("nextafter", "f16_large_out_3d", _make_binary(ntops.torch.nextafter, torch.nextafter, (256, 256, 256), torch.float16, out=True)),
    PerfCase("nextafter", "f32_mid_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_MID_NUMEL,), torch.float32)),
    PerfCase("nextafter", "f16_mid_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_MID_NUMEL,), torch.float16)),
    PerfCase("nextafter", "f64_mid_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_MID_NUMEL,), torch.float64)),
    PerfCase("nextafter", "f32_small_1d", _make_binary(ntops.torch.nextafter, torch.nextafter, (_SMALL_NUMEL,), torch.float32)),
    PerfCase("nextafter", "f32_broadcast_rect_2048x8192", _make_binary_broadcast_rect(ntops.torch.nextafter, torch.nextafter, 2048, 8192, torch.float32)),
    PerfCase("nextafter", "f32_broadcast_4096", _make_binary_broadcast(ntops.torch.nextafter, torch.nextafter, 4096, torch.float32)),
    PerfCase("nextafter", "f16_noncontig_4096", _make_binary_noncontig(ntops.torch.nextafter, torch.nextafter, 4096, torch.float16)),
    PerfCase("nextafter", "f32_noncontig_4096", _make_binary_noncontig(ntops.torch.nextafter, torch.nextafter, 4096, torch.float32)),
    PerfCase("nextafter", "f64_noncontig_2048", _make_binary_noncontig(ntops.torch.nextafter, torch.nextafter, 2048, torch.float64)),
    PerfCase("nextafter", "f32_permute3d_out_256x256x128", _make_binary_permute3d(ntops.torch.nextafter, torch.nextafter, (256, 256, 128), torch.float32, out=True)),
]


def _time_cuda(fn, warmup=5, iterations=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _assert_outputs_match(output, reference):
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3, equal_nan=True)
    else:
        assert torch.equal(output, reference)


def perf_cases_for(op_name):
    return [case for case in _PERF_CASES if case.op_name == op_name]


@skip_if_cuda_not_available
def run_perf_case(case):
    ntops_call, torch_call = case.make_pair()

    ntops_output = ntops_call()
    reference = torch_call()
    _assert_outputs_match(ntops_output, reference)

    ntops_ms = _time_cuda(ntops_call)
    torch_ms = _time_cuda(torch_call)
    torch_speed_ratio = torch_ms / ntops_ms
    print(
        f"{case.op_name}/{case.case_name}: ntops={ntops_ms:.4f} ms, "
        f"torch={torch_ms:.4f} ms, torch/ntops={torch_speed_ratio:.3f}x"
    )
    assert torch_speed_ratio >= _MIN_TORCH_SPEED_RATIO

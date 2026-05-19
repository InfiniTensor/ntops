import os
import statistics

import pytest
import torch
import triton.testing as tt


_RUN_PERF_ENV = "NTOPS_RUN_PERF"


def skip_unless_perf_enabled():
    if os.environ.get(_RUN_PERF_ENV, "0") != "1":
        pytest.skip(
            f"perf benchmark; set {_RUN_PERF_ENV}=1 to run",
            allow_module_level=True,
        )


SHAPES = [
    (13, 4),
    (8, 16),
    (2, 3, 4),
    (16, 5632),
    (256, 5632),
    (1024, 5632),
]


FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
}


def dtype_name(dtype):
    return _DTYPE_NAMES.get(dtype, str(dtype))


MIN_RATIO = 0.5


def bench_us(fn, *, warmup=50, rep=200, repeat=3):
    runs = [tt.do_bench(fn, warmup=warmup, rep=rep) * 1000 for _ in range(repeat)]
    return statistics.median(runs)


def report_and_assert(op_name, shape, dtype, ntops_us, torch_us):
    ratio = torch_us / ntops_us if ntops_us > 0 else 0.0
    print(
        f"\n  {op_name:9s}  shape={str(tuple(shape)):14s} dtype={dtype_name(dtype):5s}  "
        f"ntops={ntops_us:8.2f}us  torch={torch_us:8.2f}us  ratio={ratio:.3f}",
        end="",
    )
    assert ratio >= MIN_RATIO, (
        f"{op_name} perf regression: ratio {ratio:.3f} < {MIN_RATIO} "
        f"(ntops={ntops_us:.2f}us, torch={torch_us:.2f}us, "
        f"shape={tuple(shape)}, dtype={dtype_name(dtype)})"
    )


def warmup_pair(ntops_fn, torch_fn, n=50):
    for _ in range(n):
        ntops_fn()
        torch_fn()
    torch.cuda.synchronize()


def make_float_input(shape, dtype, *, op_name=None):
    if op_name == "lgamma":
        return torch.rand(shape, dtype=dtype, device="cuda") * 5.0 + 0.5
    return torch.randn(shape, dtype=dtype, device="cuda")


def make_int_input(shape, dtype):
    info = torch.iinfo(dtype)
    lo = max(info.min, -32768)
    hi = min(info.max, 32767)
    return torch.randint(lo, hi, shape, dtype=dtype, device="cuda")

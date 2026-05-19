import itertools

import pytest
import torch

import ntops
from tests.perf_utils import (
    FLOAT_DTYPES,
    SHAPES,
    bench_us,
    dtype_name,
    make_float_input,
    report_and_assert,
    skip_unless_perf_enabled,
    warmup_pair,
)
from tests.skippers import skip_if_cuda_not_available


skip_unless_perf_enabled()


_PARAMS = list(itertools.product(SHAPES, FLOAT_DTYPES))
_IDS = [f"{tuple(s)}-{dtype_name(d)}" for s, d in _PARAMS]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape, dtype", _PARAMS, ids=_IDS)
def test_rad2deg_perf(shape, dtype):
    a = make_float_input(shape, dtype)
    out = torch.empty_like(a)

    ntops_fn = lambda: ntops.torch.rad2deg(a, out=out)
    torch_fn = lambda: torch.rad2deg(a, out=out)
    warmup_pair(ntops_fn, torch_fn)

    ntops_us = bench_us(ntops_fn)
    torch_us = bench_us(torch_fn)
    report_and_assert("rad2deg", shape, dtype, ntops_us, torch_us)

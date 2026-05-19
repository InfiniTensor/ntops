import itertools

import pytest
import torch

import ntops
from tests.perf_utils import (
    INT_DTYPES,
    SHAPES,
    bench_us,
    dtype_name,
    make_int_input,
    report_and_assert,
    skip_unless_perf_enabled,
    warmup_pair,
)
from tests.skippers import skip_if_cuda_not_available


skip_unless_perf_enabled()


_PARAMS = list(itertools.product(SHAPES, INT_DTYPES))
_IDS = [f"{tuple(s)}-{dtype_name(d)}" for s, d in _PARAMS]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape, dtype", _PARAMS, ids=_IDS)
def test_lcm_perf(shape, dtype):
    a = make_int_input(shape, dtype)
    b = make_int_input(shape, dtype)
    out = torch.empty_like(a)

    ntops_fn = lambda: ntops.torch.lcm(a, b, out=out)
    torch_fn = lambda: torch.lcm(a, b, out=out)
    warmup_pair(ntops_fn, torch_fn)

    ntops_us = bench_us(ntops_fn)
    torch_us = bench_us(torch_fn)
    report_and_assert("lcm", shape, dtype, ntops_us, torch_us)

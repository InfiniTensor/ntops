"""Shared Triton benchmark and autotune reporting helpers."""

from collections.abc import Mapping

import torch
from triton.testing import do_bench


def benchmark_mean(function):
    """Return mean latency in microseconds using Triton's benchmark policy."""
    with torch.inference_mode():
        return (
            do_bench(
                function,
                warmup=25,
                rep=100,
                return_mode="mean",
            )
            * 1000.0
        )


def selected_config(kernel, fallback=None):
    """Return the most recently cached ``(num_warps, num_stages)`` pair."""
    # The SSA-first NineToothed compiler records the selected candidate on
    # the public handle instead of exposing Triton's Autotuner object.
    candidate = getattr(kernel, "_selected_tuning_candidate", None)
    if candidate is not None:
        if isinstance(candidate, Mapping):
            return int(candidate["num_warps"]), int(candidate["num_stages"])
        return int(candidate.num_warps), int(candidate.num_stages)

    # Keep compatibility with the legacy frontend, which emits a decorated
    # Triton kernel and stores the winner in the decorator's cache.
    globals_ = getattr(getattr(kernel, "_kernel", None), "__globals__", {})
    autotuners = (
        value
        for name, value in globals_.items()
        if name.endswith("_with_auto_tuning") and hasattr(value, "cache")
    )
    autotuner = next(autotuners, None)
    if autotuner is None or not autotuner.cache:
        if fallback is not None:
            return tuple(int(value) for value in fallback)
        raise RuntimeError("the kernel has not completed autotuning")
    config = next(reversed(autotuner.cache.values()))
    return int(config.num_warps), int(config.num_stages)

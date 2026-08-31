"""Diagnose gated RMSNorm launch, preparation, and allocation overhead.

The default case is deliberately fixed to the representative configuration
used by the Track3 investigation: ``tokens=1024``, ``hidden_size=128``,
``group_size=64``, BF16 input/gate/weight, sigmoid gate after normalization.
All five paths use the same input storage and, for the direct paths, the same
grouped tensor objects and one ``_cached_make`` kernel:

``A`` public wrapper (new output and grouped views on every call)
``B`` direct kernel with one persistent output (prepared identity should hit)
``C`` direct kernel with a new ``empty_like`` output on every call
``D`` only ``empty_like`` allocation
``E`` independent PyTorch reference

This is a diagnostic, not a submission benchmark.  CUDA Events are required
for the measurements.  If the backend cannot provide event timing, the
script exits with a labelled error instead of mixing a host-timer result into
the record.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import importlib.metadata
import json
import statistics
from pathlib import Path
from typing import Callable, Optional, Sequence


_TOKENS = 1024
_HIDDEN_SIZE = 128
_GROUP_SIZE = 64
_DTYPE_NAME = "bfloat16"
_WARMUP = 20
_REPEAT = 100
_TRIALS = 3
_EPS = 1e-5
_RTOL = 1e-2
_ATOL = 1e-2


class _EventTimingUnavailable(RuntimeError):
    """The selected CUDA backend does not implement timed CUDA Events."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=20240828)
    parser.add_argument("--warmup", type=_nonnegative_int, default=_WARMUP)
    parser.add_argument("--repeat", type=_positive_int, default=_REPEAT)
    parser.add_argument("--trials", type=_positive_int, default=_TRIALS)
    parser.add_argument("--eps", type=float, default=_EPS)
    parser.add_argument(
        "--output", type=Path, help="write the single JSON record here instead of stdout"
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _version(distribution: str, module: object) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _sync(torch, device) -> None:
    torch.cuda.synchronize(device)


def _event_error(stage: str, error: BaseException) -> _EventTimingUnavailable:
    return _EventTimingUnavailable(
        "CUDA Event timing unavailable on this backend "
        f"while {stage}; no host-timer fallback is used: {type(error).__name__}: {error}"
    )


def _time_events(
    torch,
    device,
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    trials: int,
) -> list[float]:
    """Return microseconds per call, using only CUDA Event timing."""

    for _ in range(warmup):
        fn()
    _sync(torch, device)

    results: list[float] = []
    with torch.cuda.device(device):
        for _ in range(trials):
            try:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            except (AttributeError, NotImplementedError, RuntimeError) as error:
                raise _event_error("creating timing events", error) from error

            try:
                start.record()
            except (AttributeError, NotImplementedError, RuntimeError) as error:
                raise _event_error("recording the start event", error) from error

            for _ in range(repeat):
                fn()

            try:
                end.record()
            except (AttributeError, NotImplementedError, RuntimeError) as error:
                raise _event_error("recording the end event", error) from error
            _sync(torch, device)
            try:
                elapsed_ms = start.elapsed_time(end)
            except (AttributeError, NotImplementedError, RuntimeError) as error:
                raise _event_error("reading event elapsed time", error) from error
            results.append(float(elapsed_ms) * 1000.0 / repeat)
    return results


def _reference(torch, input, gate, weight, eps: float, group_size: int):
    value = input.float() * torch.sigmoid(gate.float())
    rms = torch.sqrt(
        torch.sum(value * value, dim=-1, keepdim=True) / group_size + eps
    )
    return (value / rms * weight.float()).to(input.dtype)


def _error(torch, actual, expected) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-12)
    relative = difference / denominator
    return float(difference.max().item()), float(relative.max().item())


def _check(torch, actual, expected) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "passed",
        "max_abs": None,
        "max_rel": None,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
    }
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        result["status"] = "failed_shape_or_dtype"
        result["expected_shape"] = list(expected.shape)
        result["expected_dtype"] = str(expected.dtype)
        return result

    max_abs, max_rel = _error(torch, actual, expected)
    result["max_abs"] = max_abs
    result["max_rel"] = max_rel
    if not torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL, equal_nan=False):
        result["status"] = "failed_tolerance"
    return result


def _measure_variant(
    torch,
    device,
    name: str,
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    trials: int,
) -> dict[str, object]:
    trials_us = _time_events(torch, device, fn, warmup, repeat, trials)
    return {
        "description": name,
        "trials_us_per_call": trials_us,
        "median_us_per_call": float(statistics.median(trials_us)),
    }


def _write_record(record: dict[str, object], output: Optional[Path]) -> None:
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if output is None:
        print(line)
        return
    output.write_text(line + "\n", encoding="utf-8")


def _run(args, torch, ntops, ninetoothed) -> dict[str, object]:
    if args.eps < 0:
        raise ValueError("--eps must be non-negative")
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("gated RMSNorm launch diagnosis requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is unavailable")

    device_index = (
        torch.cuda.current_device()
        if requested_device.index is None
        else requested_device.index
    )
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device)

    torch.manual_seed(args.seed)
    input = torch.randn(
        (_TOKENS, _HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    gate = torch.randn_like(input)
    weight = torch.randn((_HIDDEN_SIZE,), device=device, dtype=torch.bfloat16)

    grouped_shape = (_TOKENS, _HIDDEN_SIZE // _GROUP_SIZE, _GROUP_SIZE)
    grouped_input = input.reshape(grouped_shape)
    grouped_gate = gate.reshape(grouped_shape)
    grouped_weight = weight.reshape((1, _HIDDEN_SIZE // _GROUP_SIZE, _GROUP_SIZE))
    grouped_weight = grouped_weight.expand(grouped_shape)
    fixed_output = torch.empty_like(grouped_input)

    activation = ntops.kernels.gated_rms_norm.ActivationVariant.SIGMOID
    kernel = ntops.torch.utils._cached_make(
        ntops.kernels.gated_rms_norm.premake,
        grouped_input.ndim,
        has_gate=True,
        norm_before_gate=False,
        activation=activation,
        block_size=1 << (_GROUP_SIZE - 1).bit_length(),
    )
    same_kernel = kernel is ntops.torch.utils._cached_make(
        ntops.kernels.gated_rms_norm.premake,
        grouped_input.ndim,
        has_gate=True,
        norm_before_gate=False,
        activation=activation,
        block_size=1 << (_GROUP_SIZE - 1).bit_length(),
    )
    if not same_kernel:
        raise RuntimeError("_cached_make did not return one shared kernel object")

    def public_wrapper():
        return ntops.torch.gated_rms_norm(
            input,
            gate,
            weight,
            eps=args.eps,
            group_size=_GROUP_SIZE,
            norm_before_gate=False,
            activation="sigmoid",
        )

    def direct_fixed():
        kernel(
            grouped_input,
            grouped_gate,
            grouped_weight,
            args.eps,
            fixed_output,
            _GROUP_SIZE,
        )
        return fixed_output

    def direct_new_output():
        output = torch.empty_like(grouped_input)
        kernel(
            grouped_input,
            grouped_gate,
            grouped_weight,
            args.eps,
            output,
            _GROUP_SIZE,
        )
        return output

    def empty_only():
        return torch.empty_like(grouped_input)

    def reference():
        return _reference(torch, grouped_input, grouped_gate, grouped_weight, args.eps, _GROUP_SIZE)

    with torch.no_grad():
        expected = reference()
        public_result = public_wrapper()
        direct_fixed_result = direct_fixed()
        direct_new_result = direct_new_output()
        _sync(torch, device)

        correctness = {
            "A_public_wrapper": _check(
                torch, public_result.reshape(grouped_shape), expected
            ),
            "B_direct_fixed_output": _check(torch, direct_fixed_result, expected),
            "C_direct_new_output": _check(torch, direct_new_result, expected),
            "D_empty_only": {"status": "not_applicable"},
            "E_pytorch_reference": {"status": "reference"},
        }
        if any(
            item["status"] not in {"passed", "reference", "not_applicable"}
            for item in correctness.values()
        ):
            raise RuntimeError("one or more NineToothed paths failed correctness")

        variants = {
            "A_public_wrapper": _measure_variant(
                torch,
                device,
                "new wrapper output and grouped views per call",
                public_wrapper,
                args.warmup,
                args.repeat,
                args.trials,
            ),
            "B_direct_fixed_output": _measure_variant(
                torch,
                device,
                "direct kernel with persistent output and tensor identities",
                direct_fixed,
                args.warmup,
                args.repeat,
                args.trials,
            ),
            "C_direct_new_output": _measure_variant(
                torch,
                device,
                "direct kernel with new empty_like output per call",
                direct_new_output,
                args.warmup,
                args.repeat,
                args.trials,
            ),
            "D_empty_only": _measure_variant(
                torch,
                device,
                "only torch.empty_like(grouped_input)",
                empty_only,
                args.warmup,
                args.repeat,
                args.trials,
            ),
            "E_pytorch_reference": _measure_variant(
                torch,
                device,
                "independent PyTorch reference",
                reference,
                args.warmup,
                args.repeat,
                args.trials,
            ),
        }

    return {
        "schema_version": 1,
        "record_type": "gated_rms_norm_launch_diagnosis",
        "status": "passed",
        "started_at_utc": _utc_now(),
        "finished_at_utc": _utc_now(),
        "case": {
            "tokens": _TOKENS,
            "hidden_size": _HIDDEN_SIZE,
            "group_size": _GROUP_SIZE,
            "dtype": _DTYPE_NAME,
            "norm_before_gate": False,
            "activation": "sigmoid",
            "gate_position": "after",
            "eps": args.eps,
            "shape": list(grouped_shape),
            "block_size": 1 << (_GROUP_SIZE - 1).bit_length(),
        },
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": device_name,
        },
        "versions": {
            "torch": torch.__version__,
            "ntops": _version("ntops", ntops),
            "ninetoothed": _version("ninetoothed", ninetoothed),
        },
        "seed": args.seed,
        "protocol": {
            "timer": "cuda_event",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
            "correctness_rtol": _RTOL,
            "correctness_atol": _ATOL,
            "kernel_shared_by": ["B_direct_fixed_output", "C_direct_new_output"],
            "max_num_configs": 1,
        },
        "correctness": correctness,
        "variants": variants,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Keep this before argument parsing, tensor creation, and every operator
    # call.  The process running this script must set the search bound first.
    import ntops

    ntops.torch.utils.set_default_max_num_configs(1)

    import ninetoothed
    import torch

    try:
        args = _parser().parse_args(argv)
        record = _run(args, torch, ntops, ninetoothed)
    except (argparse.ArgumentError, ValueError, RuntimeError) as error:
        record = {
            "schema_version": 1,
            "record_type": "gated_rms_norm_launch_diagnosis",
            "status": "error",
            "finished_at_utc": _utc_now(),
            "error": f"{type(error).__name__}: {error}",
        }
        output = getattr(locals().get("args", None), "output", None)
        _write_record(record, output)
        return 1

    _write_record(record, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

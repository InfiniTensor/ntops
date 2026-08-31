"""Fixed-configuration benchmark for :func:`ntops.torch.gated_rms_norm`.

The script intentionally keeps the benchmark protocol in a fresh process.  In
particular, ``max_num_configs`` is set to one before any tensor or operator
call.  The gated RMSNorm wrapper derives and passes a next-power-of-two
``block_size`` from ``group_size``; this benchmark records that derived value
but does not tune or override it.

The default case list is a small, representative vLLM-style matrix rather
than a Cartesian product.  It covers token counts 128/1024/4096, hidden sizes
128/256/1024, full/64/128 groups, FP16/BF16, both gate orders, and both gated
and ungated calls.  Use ``--quick`` or the filters when a shorter run is
needed.  Output is JSONL by default and can be changed to CSV with
``--format csv``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _datetime
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

import ninetoothed
import ntops
import ntops.torch.utils


_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORRECTNESS_RTOL = 1e-2
_CORRECTNESS_ATOL = 1e-2
_DEFAULT_EPS = 1e-5


@dataclasses.dataclass(frozen=True)
class Case:
    """One fixed benchmark configuration."""

    case_id: str
    tokens: int
    hidden_size: int
    group_size: Optional[int]
    dtype_name: str
    norm_before_gate: bool
    activation: str
    has_gate: bool
    quick: bool = False

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_name]

    @property
    def effective_group_size(self) -> int:
        return self.hidden_size if self.group_size is None else self.group_size

    @property
    def block_size(self) -> int:
        # Keep this in sync with ntops.torch.gated_rms_norm: the wrapper
        # explicitly passes this next-power-of-two value to premake().
        group_size = self.effective_group_size
        return 1 << (group_size - 1).bit_length()


# A curated matrix keeps the default protocol useful without running every
# Cartesian combination.  Every listed dimension is represented in the set.
CASES: tuple[Case, ...] = (
    Case(
        "t128_h128_gfull_f16_after_silu",
        128,
        128,
        None,
        "float16",
        False,
        "swish",
        True,
        True,
    ),
    Case(
        "t128_h256_g64_bf16_before_silu",
        128,
        256,
        64,
        "bfloat16",
        True,
        "swish",
        True,
        False,
    ),
    Case(
        "t128_h1024_g128_f16_before_sigmoid",
        128,
        1024,
        128,
        "float16",
        True,
        "sigmoid",
        True,
        False,
    ),
    Case(
        "t1024_h128_g64_bf16_after_sigmoid",
        1024,
        128,
        64,
        "bfloat16",
        False,
        "sigmoid",
        True,
        True,
    ),
    Case(
        "t1024_h256_gfull_f16_before_silu",
        1024,
        256,
        None,
        "float16",
        True,
        "swish",
        True,
        False,
    ),
    Case(
        "t1024_h1024_g128_bf16_after_silu",
        1024,
        1024,
        128,
        "bfloat16",
        False,
        "swish",
        True,
        False,
    ),
    Case(
        "t4096_h128_g128_f16_after_silu",
        4096,
        128,
        128,
        "float16",
        False,
        "swish",
        True,
        True,
    ),
    Case(
        "t4096_h256_g64_bf16_before_sigmoid",
        4096,
        256,
        64,
        "bfloat16",
        True,
        "sigmoid",
        True,
    ),
    Case(
        "t4096_h1024_gfull_f16_before_silu",
        4096,
        1024,
        None,
        "float16",
        True,
        "swish",
        True,
        False,
    ),
    Case(
        "t128_h256_gfull_bf16_nogate",
        128,
        256,
        None,
        "bfloat16",
        False,
        "swish",
        False,
        True,
    ),
    Case(
        "t1024_h1024_g64_f16_nogate",
        1024,
        1024,
        64,
        "float16",
        False,
        "swish",
        False,
    ),
    Case(
        "t4096_h128_gfull_bf16_nogate",
        4096,
        128,
        None,
        "bfloat16",
        False,
        "swish",
        False,
    ),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parse_int_filter(value: Optional[str], option_name: str) -> Optional[set[int]]:
    if value is None:
        return None

    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{option_name} expects comma-separated integers"
            ) from error
    if not result:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty")
    return result


def _parse_group_filter(value: Optional[str]) -> Optional[set[Optional[int]]]:
    if value is None:
        return None

    result: set[Optional[int]] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"none", "full"}:
            result.add(None)
            continue
        try:
            group_size = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "--group-size expects comma-separated integers or none"
            ) from error
        if group_size <= 0:
            raise argparse.ArgumentTypeError("group sizes must be positive")
        result.add(group_size)
    if not result:
        raise argparse.ArgumentTypeError("--group-size must not be empty")
    return result


def _parse_bool_filter(value: str) -> Optional[bool]:
    normalized = value.lower()
    if normalized == "all":
        return None
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected all, true, or false")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument(
        "--seed", type=int, default=20240828, help="base seed (case seeds are derived)"
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=20,
        help="untimed warmup calls per implementation (default: 20)",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=100,
        help="calls per timed trial (default: 100)",
    )
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=3,
        help="timed trials per implementation (default: 3)",
    )
    parser.add_argument("--eps", type=float, default=_DEFAULT_EPS)
    parser.add_argument(
        "--format", choices=("jsonl", "csv"), default="jsonl", dest="output_format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write records to this file instead of stdout",
    )
    parser.add_argument(
        "--variant",
        default="ntops",
        help="reserved result label; this script currently invokes ntops only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only the small quick subset of the case library",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="case id (repeat or use comma-separated ids); default runs all cases",
    )
    parser.add_argument(
        "--dtype",
        choices=("all", "float16", "bfloat16"),
        default="all",
        help="dtype filter",
    )
    parser.add_argument("--tokens", help="comma-separated token-count filter")
    parser.add_argument("--hidden-size", help="comma-separated hidden-size filter")
    parser.add_argument("--group-size", help="comma-separated group-size filter")
    parser.add_argument(
        "--has-gate",
        type=_parse_bool_filter,
        default=None,
        metavar="{all,true,false}",
        help="has_gate filter",
    )
    parser.add_argument(
        "--norm-before-gate",
        type=_parse_bool_filter,
        default=None,
        metavar="{all,true,false}",
        help="gate-order filter",
    )
    return parser


def _select_cases(args: argparse.Namespace) -> list[Case]:
    selected: Iterable[Case] = CASES
    if args.quick:
        selected = (case for case in selected if case.quick)

    requested_case_ids: Optional[set[str]] = None
    if args.case_ids:
        requested_case_ids = {
            case_id.strip()
            for group in args.case_ids
            for case_id in group.split(",")
            if case_id.strip()
        }
        known_case_ids = {case.case_id for case in CASES}
        unknown_case_ids = requested_case_ids - known_case_ids
        if unknown_case_ids:
            raise ValueError(
                "unknown case id(s): " + ", ".join(sorted(unknown_case_ids))
            )
        selected = (case for case in selected if case.case_id in requested_case_ids)

    token_filter = _parse_int_filter(args.tokens, "--tokens")
    hidden_filter = _parse_int_filter(args.hidden_size, "--hidden-size")
    group_filter = _parse_group_filter(args.group_size)

    def matches(case: Case) -> bool:
        return (
            (args.dtype == "all" or case.dtype_name == args.dtype)
            and (token_filter is None or case.tokens in token_filter)
            and (hidden_filter is None or case.hidden_size in hidden_filter)
            and (group_filter is None or case.group_size in group_filter)
            and (args.has_gate is None or case.has_gate == args.has_gate)
            and (
                args.norm_before_gate is None
                or case.norm_before_gate == args.norm_before_gate
            )
        )

    result = [case for case in selected if matches(case)]
    if not result:
        raise ValueError("case filters selected no cases")
    return result


def _reference(
    input: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    group_size: Optional[int],
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    """Pure PyTorch reference, intentionally independent of ntops."""

    input_float = input.float()
    gate_float = None if gate is None else gate.float()
    activation_fn: Callable[[torch.Tensor], torch.Tensor] = (
        torch.sigmoid if activation == "sigmoid" else F.silu
    )

    if gate_float is not None and not norm_before_gate:
        input_float = input_float * activation_fn(gate_float)

    effective_group_size = group_size or input.shape[-1]
    grouped = input_float.reshape(*input.shape[:-1], -1, effective_group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    output = grouped * torch.rsqrt(variance + eps)
    output = output.reshape_as(input_float) * weight.float()

    if gate_float is not None and norm_before_gate:
        output = output * activation_fn(gate_float)

    return output.to(input.dtype)


def _case_seed(base_seed: int, case_index: int) -> int:
    # Avoid Python's randomized hash so case data remains reproducible across
    # fresh processes.
    return (base_seed + 1009 * (case_index + 1)) % (2**63 - 1)


def _make_inputs(
    case: Case, device: torch.device, seed: int
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    torch.manual_seed(seed)
    input = torch.randn(
        (case.tokens, case.hidden_size), device=device, dtype=case.dtype
    )
    gate = (
        torch.randn_like(input)
        if case.has_gate
        else None
    )
    weight = torch.randn((case.hidden_size,), device=device, dtype=case.dtype)
    return input, gate, weight


def _call_ntops(
    case: Case,
    input: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Do not pass a competing block size here: the wrapper explicitly computes
    # next_power_of_two(group_size) and supplies it to the NineToothed premake.
    return ntops.torch.gated_rms_norm(
        input,
        gate,
        weight,
        eps=eps,
        group_size=case.group_size,
        norm_before_gate=case.norm_before_gate,
        activation=case.activation,
    )


def _cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _time_trials_cuda_events(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
    trials: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    _cuda_sync(device)

    results: list[float] = []
    with torch.cuda.device(device):
        for _ in range(trials):
            try:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            except (NotImplementedError, RuntimeError) as error:
                raise _EventTimingUnavailable(error) from error
            for _ in range(repeat):
                fn()
            try:
                end.record()
            except (NotImplementedError, RuntimeError) as error:
                raise _EventTimingUnavailable(error) from error
            _cuda_sync(device)
            try:
                elapsed_ms = start.elapsed_time(end)
            except (NotImplementedError, RuntimeError) as error:
                raise _EventTimingUnavailable(error) from error
            results.append(elapsed_ms * 1000.0 / repeat)
    return results


class _EventTimingUnavailable(Exception):
    """Internal signal used only to select the labelled host-timer fallback."""


def _time_trials_perf_counter(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
    trials: int,
) -> list[float]:
    """Portable fallback for backends whose CUDA Event lacks timing support."""

    for _ in range(warmup):
        fn()
    _cuda_sync(device)

    results: list[float] = []
    for _ in range(trials):
        _cuda_sync(device)
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        _cuda_sync(device)
        results.append((time.perf_counter() - start) * 1e6 / repeat)
    return results


def _time_trials(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[list[float], str]:
    try:
        return (
            _time_trials_cuda_events(fn, device, warmup, repeat, trials),
            "cuda_event",
        )
    except _EventTimingUnavailable as error:
        # Event timing is preferred, but some non-NVIDIA CUDA backends expose
        # only synchronization.  Do not fabricate a result: the fallback is
        # explicitly labelled in every record.
        print(
            f"warning: CUDA Event timing unavailable; using perf_counter ({error})",
            file=sys.stderr,
        )
        return (
            _time_trials_perf_counter(fn, device, warmup, repeat, trials),
            "perf_counter",
        )


def _package_version(distribution: str, module: object) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _git_description(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _max_error(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-12)
    relative = difference / denominator
    return float(difference.max().item()), float(relative.max().item())


_OUTPUT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "record_type",
    "status",
    "variant",
    "started_at_utc",
    "finished_at_utc",
    "device",
    "device_name",
    "torch_version",
    "ntops_version",
    "ninetoothed_version",
    "ntops_git",
    "seed",
    "case_index",
    "case_id",
    "tokens",
    "hidden_size",
    "shape",
    "dtype",
    "group_size",
    "effective_group_size",
    "block_size",
    "norm_before_gate",
    "activation",
    "has_gate",
    "eps",
    "correctness_rtol",
    "correctness_atol",
    "correctness_max_abs",
    "correctness_max_rel",
    "warmup",
    "repeat",
    "trials",
    "timer",
    "ntops_trials_us",
    "ntops_median_us",
    "reference_trials_us",
    "reference_median_us",
    "speedup_reference_over_ntops",
    "error",
)


def _base_record(
    case: Case,
    case_index: int,
    case_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    device_name: str,
    run_started_at: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "gated_rms_norm_benchmark",
        "status": "error",
        "variant": args.variant,
        "started_at_utc": run_started_at,
        "finished_at_utc": None,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "ntops_version": _package_version("ntops", ntops),
        "ninetoothed_version": _package_version("ninetoothed", ninetoothed),
        "ntops_git": _git_description(_SCRIPT_REPO_ROOT),
        "seed": case_seed,
        "case_index": case_index,
        "case_id": case.case_id,
        "tokens": case.tokens,
        "hidden_size": case.hidden_size,
        "shape": [case.tokens, case.hidden_size],
        "dtype": case.dtype_name,
        "group_size": case.group_size,
        "effective_group_size": case.effective_group_size,
        "block_size": case.block_size,
        "norm_before_gate": case.norm_before_gate,
        "activation": case.activation,
        "has_gate": case.has_gate,
        "eps": args.eps,
        "correctness_rtol": _CORRECTNESS_RTOL,
        "correctness_atol": _CORRECTNESS_ATOL,
        "correctness_max_abs": None,
        "correctness_max_rel": None,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "timer": None,
        "ntops_trials_us": None,
        "ntops_median_us": None,
        "reference_trials_us": None,
        "reference_median_us": None,
        "speedup_reference_over_ntops": None,
        "error": None,
    }


def _benchmark_case(
    case: Case,
    case_index: int,
    args: argparse.Namespace,
    device: torch.device,
    device_name: str,
    run_started_at: str,
) -> dict[str, object]:
    case_seed = _case_seed(args.seed, case_index)
    record = _base_record(
        case, case_index, case_seed, args, device, device_name, run_started_at
    )

    try:
        input, gate, weight = _make_inputs(case, device, case_seed)
        with torch.no_grad():
            expected = _reference(
                input,
                gate,
                weight,
                args.eps,
                case.group_size,
                case.norm_before_gate,
                case.activation,
            )
            actual = _call_ntops(case, input, gate, weight, args.eps)
        _cuda_sync(device)

        max_abs, max_rel = _max_error(actual, expected)
        record["correctness_max_abs"] = max_abs
        record["correctness_max_rel"] = max_rel
        if not torch.allclose(
            actual,
            expected,
            rtol=_CORRECTNESS_RTOL,
            atol=_CORRECTNESS_ATOL,
            equal_nan=False,
        ):
            record["status"] = "correctness_failed"
            record["error"] = (
                "ntops output did not meet torch reference tolerance "
                f"(rtol={_CORRECTNESS_RTOL}, atol={_CORRECTNESS_ATOL})"
            )
            return record

        ntops_fn = lambda: _call_ntops(case, input, gate, weight, args.eps)
        reference_fn = lambda: _reference(
            input,
            gate,
            weight,
            args.eps,
            case.group_size,
            case.norm_before_gate,
            case.activation,
        )

        # Each implementation is warmed up and timed independently.  A
        # correctness failure returns above, so failed cases are never timed.
        ntops_trials, timer = _time_trials(
            ntops_fn, device, args.warmup, args.repeat, args.trials
        )
        reference_trials, reference_timer = _time_trials(
            reference_fn, device, args.warmup, args.repeat, args.trials
        )
        if timer != reference_timer:
            timer = f"{timer}+{reference_timer}"

        ntops_median = float(statistics.median(ntops_trials))
        reference_median = float(statistics.median(reference_trials))
        record.update(
            {
                "status": "passed",
                "timer": timer,
                "ntops_trials_us": ntops_trials,
                "ntops_median_us": ntops_median,
                "reference_trials_us": reference_trials,
                "reference_median_us": reference_median,
                "speedup_reference_over_ntops": reference_median / ntops_median,
            }
        )
    except Exception as error:  # preserve one machine-readable record per case
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["finished_at_utc"] = _utc_now()
    return record


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_records(records: Sequence[dict[str, object]], args: argparse.Namespace) -> None:
    output = sys.stdout
    should_close = False
    if args.output is not None:
        output = args.output.open("w", newline="", encoding="utf-8")
        should_close = True

    try:
        if args.output_format == "csv":
            writer = csv.DictWriter(
                output,
                fieldnames=_OUTPUT_FIELDS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow({key: _csv_value(record.get(key)) for key in _OUTPUT_FIELDS})
        else:
            for record in records:
                output.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        output.flush()
    finally:
        if should_close:
            output.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    # This must remain the first runtime action.  In a fresh process it fixes
    # the NineToothed search space before any ntops operator is called.
    ntops.torch.utils.set_default_max_num_configs(1)

    args = _parser().parse_args(argv)
    try:
        cases = _select_cases(args)
    except (argparse.ArgumentTypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.eps < 0:
        print("error: --eps must be non-negative", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        print("error: gated RMSNorm benchmark requires a CUDA device", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("error: torch.cuda is unavailable", file=sys.stderr)
        return 2

    device_index = torch.cuda.current_device() if device.index is None else device.index
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device)
    run_started_at = _utc_now()

    records: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        print(
            f"[{case_index + 1}/{len(cases)}] {case.case_id}",
            file=sys.stderr,
            flush=True,
        )
        records.append(
            _benchmark_case(
                case,
                case_index,
                args,
                device,
                device_name,
                run_started_at,
            )
        )

    _write_records(records, args)
    return 0 if all(record["status"] == "passed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

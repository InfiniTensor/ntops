#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_kernelswift_t2.sh [options]

Options:
  --ninetoothed-dir PATH   NineToothed source tree (default: ../ninetoothed)
  --output-dir PATH        Logs, caches, and CSV output directory
  --mode MODE              all|correctness|regression|benchmark|environment
  --baseline NAME          pytorch|vllm_triton|vllm_vendor|vllm_portable
                           (default: vllm_portable)
  --warmup N               Warm-up calls per measurement (default: 20)
  --iterations N           Timed calls per measurement (default: 100)
  --rounds N               Independent measurement rounds (default: 3)
  -h, --help               Show this help

The script imports both repositories directly through PYTHONPATH. It does not
replace the vendor PyTorch, Triton, or vLLM installation. The
`vllm_portable` option is a separately labelled vLLM-derived compatibility
baseline; it is not an unmodified upstream vLLM binary.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NTOPS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NINETOOTHED_DIR="${NINETOOTHED_DIR:-${NTOPS_DIR}/../ninetoothed}"
OUTPUT_DIR="${OUTPUT_DIR:-${NTOPS_DIR}/t2_results/run_$(date +%Y%m%d_%H%M%S)}"
MODE="all"
BASELINE="vllm_portable"
WARMUP=20
ITERATIONS=100
ROUNDS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ninetoothed-dir)
            NINETOOTHED_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${MODE}" in
    all|correctness|regression|benchmark|environment) ;;
    *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 2
        ;;
esac

case "${BASELINE}" in
    pytorch|vllm_triton|vllm_vendor|vllm_portable) ;;
    *)
        echo "Unsupported baseline: ${BASELINE}" >&2
        exit 2
        ;;
esac

require_positive_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
}

require_positive_integer --warmup "${WARMUP}"
require_positive_integer --iterations "${ITERATIONS}"
require_positive_integer --rounds "${ROUNDS}"

NINETOOTHED_DIR="$(cd -- "${NINETOOTHED_DIR}" && pwd)"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"

export PYTHONPATH="${NTOPS_DIR}/src:${NINETOOTHED_DIR}/src:${PYTHONPATH:-}"

record_repository() {
    local name="$1"
    local directory="$2"

    echo "${name}_path=${directory}"
    if git -C "${directory}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "${name}_head=$(git -C "${directory}" rev-parse HEAD)"
        echo "${name}_branch=$(git -C "${directory}" branch --show-current)"
        if [[ -n "$(git -C "${directory}" status --porcelain)" ]]; then
            echo "${name}_dirty=true"
        else
            echo "${name}_dirty=false"
        fi
    else
        echo "${name}_head=unavailable-source-snapshot"
        echo "${name}_branch=unavailable-source-snapshot"
        echo "${name}_dirty=unknown"
    fi
}

record_environment() {
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"
        uname -a
        echo "mode=${MODE}"
        echo "baseline=${BASELINE}"
        echo "warmup=${WARMUP}"
        echo "iterations=${ITERATIONS}"
        echo "rounds=${ROUNDS}"
        record_repository ntops "${NTOPS_DIR}"
        record_repository ninetoothed "${NINETOOTHED_DIR}"
        python - <<'PY'
import platform

import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_hip={torch.version.hip}")

try:
    import triton
except Exception as error:  # noqa: BLE001
    print(f"triton_error={type(error).__name__}: {error}")
else:
    print(f"triton={triton.__version__}")

try:
    import vllm
except Exception as error:  # noqa: BLE001
    print(f"vllm_error={type(error).__name__}: {error}")
else:
    print(f"vllm={vllm.__version__}")

print(f"accelerator_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"accelerator_name={properties.name}")
    print(f"accelerator_memory={properties.total_memory}")
    print(f"accelerator_warp_size={getattr(properties, 'warp_size', None)}")
    print(f"accelerator_arch={getattr(properties, 'gcnArchName', None)}")
PY
        printf 'sha256  %s\n' "T2 operator, evidence, and compiler files"
        sha256sum \
            "${NTOPS_DIR}/src/ntops/kernels/scaled_mm.py" \
            "${NTOPS_DIR}/src/ntops/torch/scaled_mm.py" \
            "${NTOPS_DIR}/tests/test_scaled_mm.py" \
            "${NTOPS_DIR}/benchmarks/probe_official_scaled_mm.py" \
            "${NTOPS_DIR}/benchmarks/vllm_portable_scaled_mm.py" \
            "${NTOPS_DIR}/scripts/run_kernelswift_t2.sh" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/base.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/ssa.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/materializers/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/compiler/runtime.py" \
            "${NINETOOTHED_DIR}/tests/test_triton_fp8_dot_fallback.py"
    } 2>&1 | tee "${OUTPUT_DIR}/environment.log"
}

run_correctness() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_correctness"
    export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton_cache_correctness"
    mkdir -p "${NINETOOTHED_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    (
        cd "${NTOPS_DIR}"
        python -m pytest tests/test_scaled_mm.py -q
    ) 2>&1 | tee "${OUTPUT_DIR}/correctness.log"
}

run_regression() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_regression"
    export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton_cache_regression"
    mkdir -p "${NINETOOTHED_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    (
        cd "${NINETOOTHED_DIR}"
        python -m pytest -q \
            tests/test_triton_fp8_dot_fallback.py \
            tests/test_ssa_first_backend_lowering.py \
            tests/test_triton_runtime_auto_tuning.py \
            tests/test_ssa_scalar_argument_emission.py \
            tests/test_static_layout_bounds.py \
            tests/test_static_tensor_strides.py
    ) 2>&1 | tee "${OUTPUT_DIR}/compiler_regression.log"
}

run_probe() {
    local candidate="$1"
    local dtype="$2"
    local m="$3"
    local round="$4"
    local log_file="${OUTPUT_DIR}/benchmark_raw/${dtype}_m${m}_round${round}_${candidate}.log"

    (
        cd "${NTOPS_DIR}"
        python benchmarks/probe_official_scaled_mm.py \
            --mode "${candidate}" \
            --m "${m}" \
            --n 4096 \
            --k 4096 \
            --fp8-dtype "${dtype}" \
            --warmup "${WARMUP}" \
            --iterations "${ITERATIONS}"
    ) 2>&1 | tee "${log_file}"
}

aggregate_benchmark() {
    python - "${OUTPUT_DIR}/benchmark_raw" "${OUTPUT_DIR}/benchmark.csv" \
        "${BASELINE}" "${ROUNDS}" <<'PY'
import csv
import pathlib
import re
import statistics
import sys


raw_dir = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
baseline_name = sys.argv[3]
expected_rounds = int(sys.argv[4])
filename = re.compile(
    r"(?P<dtype>e4m3fn|e5m2)_m(?P<m>\d+)_round(?P<round>\d+)_"
    r"(?P<candidate>[A-Za-z0-9_]+)\.log"
)
latency = re.compile(
    r"stage=latency (?:submission|baseline)=[A-Za-z0-9_]+ "
    r"latency_ms=(?P<latency>[0-9.]+)"
)
groups = {}

for path in sorted(raw_dir.glob("*.log")):
    match = filename.fullmatch(path.name)
    if match is None:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    if "stage=correctness passed=True" not in text:
        raise SystemExit(f"Correctness did not pass: {path}")
    latency_match = latency.search(text)
    if latency_match is None:
        raise SystemExit(f"Latency is missing: {path}")
    key = (match["dtype"], int(match["m"]))
    groups.setdefault(key, {}).setdefault(match["candidate"], []).append(
        float(latency_match["latency"])
    )

fieldnames = (
    "fp8_dtype",
    "m",
    "n",
    "k",
    "scaling_layout",
    "out_dtype",
    "baseline",
    "rounds",
    "submission_median_ms",
    "baseline_median_ms",
    "speedup",
    "correctness",
)
rows = []

for (dtype, m), candidates in sorted(groups.items()):
    submission = candidates.get("ntops", [])
    baseline = candidates.get(baseline_name, [])
    if len(submission) != expected_rounds or len(baseline) != expected_rounds:
        raise SystemExit(
            f"Incomplete rounds for dtype={dtype}, m={m}: "
            f"submission={len(submission)}, baseline={len(baseline)}"
        )
    submission_median = statistics.median(submission)
    baseline_median = statistics.median(baseline)
    rows.append(
        {
            "fp8_dtype": dtype,
            "m": m,
            "n": 4096,
            "k": 4096,
            "scaling_layout": "1x128-128x128",
            "out_dtype": "bfloat16",
            "baseline": baseline_name,
            "rounds": expected_rounds,
            "submission_median_ms": f"{submission_median:.6f}",
            "baseline_median_ms": f"{baseline_median:.6f}",
            "speedup": f"{baseline_median / submission_median:.6f}",
            "correctness": "pass",
        }
    )

if len(rows) != 6:
    raise SystemExit(f"Expected six benchmark rows, found {len(rows)}")

with csv_path.open("w", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        "stage=summary "
        f"dtype={row['fp8_dtype']} m={row['m']} "
        f"submission_ms={row['submission_median_ms']} "
        f"baseline_ms={row['baseline_median_ms']} "
        f"speedup={row['speedup']} correctness=pass"
    )
PY
}

run_benchmark() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_benchmark"
    export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton_cache_benchmark"
    mkdir -p \
        "${NINETOOTHED_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${OUTPUT_DIR}/benchmark_raw"

    for dtype in e4m3fn e5m2; do
        for m in 1 16 128; do
            for round in $(seq 1 "${ROUNDS}"); do
                if (( round % 2 == 1 )); then
                    run_probe ntops "${dtype}" "${m}" "${round}"
                    run_probe "${BASELINE}" "${dtype}" "${m}" "${round}"
                else
                    run_probe "${BASELINE}" "${dtype}" "${m}" "${round}"
                    run_probe ntops "${dtype}" "${m}" "${round}"
                fi
            done
        done
    done

    aggregate_benchmark 2>&1 | tee "${OUTPUT_DIR}/benchmark_summary.log"
}

record_environment

case "${MODE}" in
    all)
        run_correctness
        run_regression
        run_benchmark
        ;;
    correctness) run_correctness ;;
    regression) run_regression ;;
    benchmark) run_benchmark ;;
    environment) ;;
esac

echo "T2 run complete: ${OUTPUT_DIR}"

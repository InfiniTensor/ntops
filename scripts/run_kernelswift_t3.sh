#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_kernelswift_t3.sh [options]

Options:
  --ninetoothed-dir PATH   NineToothed source tree (default: ../ninetoothed)
  --output-dir PATH        Logs, caches, and CSV output directory
  --mode MODE              all|correctness|regression|benchmark|ablation|environment
  -h, --help               Show this help

The script imports both repositories directly through PYTHONPATH. It does not
replace the vendor PyTorch, Triton, or vLLM installation.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NTOPS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NINETOOTHED_DIR="${NINETOOTHED_DIR:-${NTOPS_DIR}/../ninetoothed}"
OUTPUT_DIR="${OUTPUT_DIR:-${NTOPS_DIR}/t3_results/run_$(date +%Y%m%d_%H%M%S)}"
MODE="all"

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
    all|correctness|regression|benchmark|ablation|environment) ;;
    *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 2
        ;;
esac

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
        printf 'sha256  %s\n' "T3 and compiler files"
        sha256sum \
            "${NTOPS_DIR}/src/ntops/kernels/rms_norm_gated.py" \
            "${NTOPS_DIR}/src/ntops/torch/rms_norm_gated.py" \
            "${NTOPS_DIR}/tests/test_rms_norm_gated.py" \
            "${NTOPS_DIR}/benchmarks/benchmark_rms_norm_gated.py" \
            "${NTOPS_DIR}/benchmarks/benchmark_rms_norm_gated_runtime_ablation.py" \
            "${NTOPS_DIR}/scripts/run_kernelswift_t3.sh" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/ssa.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/materializers/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/compiler/runtime.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/frontend/layout.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/tensor.py"
    } 2>&1 | tee "${OUTPUT_DIR}/environment.log"
}

run_correctness() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_correctness"
    mkdir -p "${NINETOOTHED_CACHE_DIR}"
    (
        cd "${NTOPS_DIR}"
        python -m pytest tests/test_rms_norm_gated.py -q
    ) 2>&1 | tee "${OUTPUT_DIR}/correctness.log"
}

run_regression() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_regression"
    mkdir -p "${NINETOOTHED_CACHE_DIR}"
    (
        cd "${NINETOOTHED_DIR}"
        python -m pytest -q \
            tests/test_ssa_first_backend_lowering.py \
            tests/test_triton_runtime_auto_tuning.py \
            tests/test_ssa_scalar_argument_emission.py \
            tests/test_static_layout_bounds.py \
            tests/test_static_tensor_strides.py
    ) 2>&1 | tee "${OUTPUT_DIR}/compiler_regression.log"
}

run_benchmark() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_benchmark"
    mkdir -p "${NINETOOTHED_CACHE_DIR}"
    (
        cd "${NTOPS_DIR}"
        python benchmarks/benchmark_rms_norm_gated.py \
            --csv "${OUTPUT_DIR}/benchmark.csv"
    ) 2>&1 | tee "${OUTPUT_DIR}/benchmark.log"
}

run_ablation() {
    export NINETOOTHED_CACHE_DIR="${OUTPUT_DIR}/cache_ablation"
    mkdir -p "${NINETOOTHED_CACHE_DIR}"
    (
        cd "${NTOPS_DIR}"
        python benchmarks/benchmark_rms_norm_gated_runtime_ablation.py
    ) 2>&1 | tee "${OUTPUT_DIR}/runtime_ablation.log"
}

record_environment

case "${MODE}" in
    all)
        run_correctness
        run_regression
        run_benchmark
        run_ablation
        ;;
    correctness) run_correctness ;;
    regression) run_regression ;;
    benchmark) run_benchmark ;;
    ablation) run_ablation ;;
    environment) ;;
esac

echo "T3 run complete: ${OUTPUT_DIR}"

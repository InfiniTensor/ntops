#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_kernelswift_t1.sh [options]

Options:
  --ninetoothed-dir PATH   NineToothed source tree (default: ../ninetoothed)
  --output-dir PATH        Logs, caches, and CSV output directory
  --mode MODE              all|environment|lint|correctness|regression|benchmark
  -h, --help               Show this help

The script imports both repositories directly through PYTHONPATH. It does not
replace the vendor PyTorch, Triton, or vLLM installation. Use a new output
directory for every clean evaluation so JIT products remain isolated.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NTOPS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NINETOOTHED_DIR="${NINETOOTHED_DIR:-${NTOPS_DIR}/../ninetoothed}"
OUTPUT_DIR="${OUTPUT_DIR:-${NTOPS_DIR}/t1_results/run_$(date +%Y%m%d_%H%M%S)}"
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
    all|environment|lint|correctness|regression|benchmark) ;;
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
        printf 'sha256  %s\n' "T1 and compiler files"
        sha256sum \
            "${NTOPS_DIR}/src/ntops/kernels/mxfp4_grouped_gemm.py" \
            "${NTOPS_DIR}/src/ntops/torch/mxfp4_grouped_gemm.py" \
            "${NTOPS_DIR}/tests/test_mxfp4_grouped_gemm.py" \
            "${NTOPS_DIR}/benchmarks/benchmark_mxfp4_grouped_gemm_reference.py" \
            "${NTOPS_DIR}/scripts/run_kernelswift_t1.sh" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/ssa.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/emitters/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/backends/materializers/triton.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/frontend/python.py" \
            "${NINETOOTHED_DIR}/src/ninetoothed/frontend/types.py"
    } 2>&1 | tee "${OUTPUT_DIR}/environment.log"
}

prepare_cache() {
    local name="$1"
    local cache_dir="${OUTPUT_DIR}/cache_${name}"

    mkdir -p "${cache_dir}/ninetoothed" "${cache_dir}/triton" "${cache_dir}/xdg"
    export NINETOOTHED_CACHE_DIR="${cache_dir}/ninetoothed"
    export TRITON_CACHE_DIR="${cache_dir}/triton"
    export XDG_CACHE_HOME="${cache_dir}/xdg"
}

run_lint() {
    {
        cd "${NINETOOTHED_DIR}"
        ruff format --check \
            src/ninetoothed/backends/emitters/base.py \
            src/ninetoothed/backends/emitters/ssa.py \
            src/ninetoothed/backends/emitters/triton.py \
            src/ninetoothed/backends/materializers/triton.py \
            src/ninetoothed/frontend/python.py \
            src/ninetoothed/frontend/types.py \
            tests/test_ssa_validation.py \
            tests/test_triton_runtime_auto_tuning.py \
            tests/test_triton_emitter_target.py
        ruff check \
            src/ninetoothed/backends/emitters/base.py \
            src/ninetoothed/backends/emitters/ssa.py \
            src/ninetoothed/backends/emitters/triton.py \
            src/ninetoothed/backends/materializers/triton.py \
            src/ninetoothed/frontend/python.py \
            src/ninetoothed/frontend/types.py \
            tests/test_ssa_validation.py \
            tests/test_triton_runtime_auto_tuning.py \
            tests/test_triton_emitter_target.py
        python scripts/check_contributing_style.py

        cd "${NTOPS_DIR}"
        ruff format --check \
            src/ntops/kernels/mxfp4_grouped_gemm.py \
            src/ntops/torch/mxfp4_grouped_gemm.py \
            tests/test_mxfp4_grouped_gemm.py \
            benchmarks/benchmark_mxfp4_grouped_gemm_reference.py
        ruff check \
            src/ntops/kernels/mxfp4_grouped_gemm.py \
            src/ntops/torch/mxfp4_grouped_gemm.py \
            tests/test_mxfp4_grouped_gemm.py \
            benchmarks/benchmark_mxfp4_grouped_gemm_reference.py
    } 2>&1 | tee "${OUTPUT_DIR}/lint.log"
}

run_correctness() {
    prepare_cache correctness
    (
        cd "${NTOPS_DIR}"
        python -m pytest tests/test_mxfp4_grouped_gemm.py -q
    ) 2>&1 | tee "${OUTPUT_DIR}/correctness.log"
}

run_regression() {
    prepare_cache regression
    (
        cd "${NINETOOTHED_DIR}"
        python -m pytest -q \
            tests/test_ssa_first_backend_lowering.py \
            tests/test_triton_runtime_auto_tuning.py \
            tests/test_ssa_validation.py \
            tests/test_triton_emitter_target.py
    ) 2>&1 | tee "${OUTPUT_DIR}/compiler_regression.log"
}

run_benchmark() {
    prepare_cache benchmark
    local benchmark="${NTOPS_DIR}/benchmarks/benchmark_mxfp4_grouped_gemm_reference.py"
    local csv="${OUTPUT_DIR}/benchmark.csv"

    (
        cd "${NTOPS_DIR}"
        python "${benchmark}" --experts 1 --n 16 --k 32 --counts 1 \
            --warmup 50 --iterations 500 --rounds 9 --csv "${csv}"
        python "${benchmark}" --experts 2 --n 32 --k 64 --counts 1,3 \
            --warmup 50 --iterations 500 --rounds 9 --csv "${csv}"
        python "${benchmark}" --experts 2 --n 64 --k 128 --counts 5,2 \
            --warmup 50 --iterations 500 --rounds 9 --csv "${csv}"
        python "${benchmark}" --experts 4 --n 128 --k 256 --counts 1,0,17,5 \
            --warmup 50 --iterations 500 --rounds 9 --csv "${csv}"
        python "${benchmark}" --experts 4 --n 256 --k 256 --counts 1,3,5,7 \
            --warmup 50 --iterations 500 --rounds 9 --csv "${csv}"
    ) 2>&1 | tee "${OUTPUT_DIR}/benchmark.log"
}

record_environment

case "${MODE}" in
    all)
        run_lint
        run_correctness
        run_regression
        run_benchmark
        ;;
    environment) ;;
    lint) run_lint ;;
    correctness) run_correctness ;;
    regression) run_regression ;;
    benchmark) run_benchmark ;;
esac

echo "T1 run complete: ${OUTPUT_DIR}"

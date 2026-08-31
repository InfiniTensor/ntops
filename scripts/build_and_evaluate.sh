#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/artifacts}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"

TEST_FILES=(
  tests/test_block_scaled_fp8_mm.py
  tests/test_mxfp4_w4a16_grouped_mm.py
  tests/test_rms_norm_gated.py
  tests/test_fused_mla_rope_cache_write.py
)

BENCHMARK_FILES=(
  benchmarks/bench_block_scaled_fp8_mm.py
  benchmarks/bench_mxfp4_w4a16_grouped_mm.py
  benchmarks/bench_rms_norm_gated.py
  benchmarks/bench_fused_mla_rope_cache_write.py
)

usage() {
  echo "usage: $0 [all|build|test|benchmark]" >&2
}

case "${MODE}" in
  all|build|test|benchmark) ;;
  *)
    usage
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -e ".[testing]" 2>&1 \
    | tee "${RESULT_DIR}/build.log"
fi

"${PYTHON_BIN}" - <<'PY' | tee "${RESULT_DIR}/environment.txt"
import inspect
import platform

import ninetoothed
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("hip", torch.version.hip)
print("ninetoothed", inspect.getfile(ninetoothed))
print("accelerator_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name())
PY

if [[ "${MODE}" == "build" ]]; then
  echo "build artifacts: ${RESULT_DIR}"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("a CUDA/HIP/CoreX-compatible accelerator is required")
PY

if [[ "${MODE}" == "all" || "${MODE}" == "test" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m pytest -q "${TEST_FILES[@]}" 2>&1 \
    | tee "${RESULT_DIR}/pytest.log"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "benchmark" ]]; then
  for benchmark_file in "${BENCHMARK_FILES[@]}"; do
    benchmark_name="$(basename "${benchmark_file}" .py)"
    PYTHONPATH=src "${PYTHON_BIN}" "${benchmark_file}" 2>&1 \
      | tee "${RESULT_DIR}/${benchmark_name}.csv"
  done
fi

echo "evaluation artifacts: ${RESULT_DIR}"

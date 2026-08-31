#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NTOPS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NINETOOTHED_ROOT="${NINETOOTHED_ROOT:-${NTOPS_ROOT}/../ninetoothed}"
NINETOOTHED_ROOT="$(cd -- "${NINETOOTHED_ROOT}" && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${1:-${NTOPS_ROOT}/results/kernelswift_t4_${timestamp}}"

if [[ -e "${RESULT_DIR}" ]]; then
    echo "结果目录已存在，拒绝复用缓存或覆盖日志：${RESULT_DIR}" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/ninetoothed-cache"

export PYTHONPATH="${NTOPS_ROOT}/src:${NINETOOTHED_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NINETOOTHED_CACHE_DIR="${RESULT_DIR}/ninetoothed-cache"

is_corex=0
if python -c 'import torch; raise SystemExit(0 if getattr(torch, "corex", False) else 1)'; then
    is_corex=1
fi

if [[ "${is_corex}" == 1 ]]; then
    export NINETOOTHED_BACKEND="${NINETOOTHED_BACKEND:-cuda}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/corex}"
    export NINETOOTHED_CUDA_COMPILER="${NINETOOTHED_CUDA_COMPILER:-${CUDA_HOME}/bin/clang++}"
    export NINETOOTHED_CUDA_LANGUAGE="${NINETOOTHED_CUDA_LANGUAGE:-ivcore}"
else
    export NINETOOTHED_BACKEND="${NINETOOTHED_BACKEND:-triton}"
fi

python - <<'PY' | tee "${RESULT_DIR}/logs/environment.log"
import platform

import torch
import triton
import vllm

print("python", platform.python_version())
print("torch", torch.__version__)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("corex", bool(getattr(torch, "corex", False)))
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("没有可用的 CUDA 兼容 GPU")
print("device", torch.cuda.get_device_name(0))
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("gpu_health", x.item())
PY

(
    cd "${NTOPS_ROOT}"
    python -m ruff check \
        benchmarks/benchmark_mla_rope_cache.py \
        src/ntops/kernels/mla_rope_cache.py \
        src/ntops/kernels/mla_rope_cache_pair.py \
        src/ntops/torch/mla_rope_cache.py \
        src/ntops/torch/mla_rope_cache_pair.py \
        tests/test_mla_rope_cache.py
) 2>&1 | tee "${RESULT_DIR}/logs/ruff_ntops.log"

(
    cd "${NINETOOTHED_ROOT}"
    python -m ruff check \
        src/ninetoothed/backends/emitters/ssa.py \
        src/ninetoothed/backends/materializers/cuda.py \
        src/ninetoothed/backends/toolchain.py \
        tests/test_compiler_cache_runtime.py \
        tests/test_backend_registry.py
) 2>&1 | tee "${RESULT_DIR}/logs/ruff_ninetoothed.log"

(
    cd "${NTOPS_ROOT}"
    python -m pytest -q tests/test_mla_rope_cache.py
) 2>&1 | tee "${RESULT_DIR}/logs/correctness.log"

(
    cd "${NINETOOTHED_ROOT}"
    python -m pytest -q \
        tests/test_ssa_first_backend_lowering.py \
        tests/test_triton_runtime_auto_tuning.py \
        tests/test_ssa_scalar_argument_emission.py \
        tests/test_static_layout_bounds.py \
        tests/test_static_tensor_strides.py \
        tests/test_compiler_cache_runtime.py \
        tests/test_backend_registry.py
) 2>&1 | tee "${RESULT_DIR}/logs/compiler_regression.log"

for heads in 32 128; do
    (
        cd "${NTOPS_ROOT}"
        python benchmarks/benchmark_mla_rope_cache.py --num-heads "${heads}"
    ) 2>&1 | tee "${RESULT_DIR}/logs/benchmark_heads${heads}.log"
done

(
    cd "${NTOPS_ROOT}"
    sha256sum \
        src/ntops/kernels/mla_rope_cache.py \
        src/ntops/kernels/mla_rope_cache_pair.py \
        src/ntops/torch/mla_rope_cache.py \
        src/ntops/torch/mla_rope_cache_pair.py \
        benchmarks/benchmark_mla_rope_cache.py \
        tests/test_mla_rope_cache.py
    cd "${NINETOOTHED_ROOT}"
    sha256sum \
        src/ninetoothed/backends/emitters/ssa.py \
        src/ninetoothed/backends/materializers/cuda.py \
        src/ninetoothed/backends/toolchain.py \
        tests/test_compiler_cache_runtime.py \
        tests/test_backend_registry.py
) > "${RESULT_DIR}/SHA256SUMS"

grep -hE 'shape=.*correctness=pass' \
    "${RESULT_DIR}/logs/benchmark_heads32.log" \
    "${RESULT_DIR}/logs/benchmark_heads128.log" \
    | tee "${RESULT_DIR}/FORMAL_SPEEDUP_SUMMARY"

echo "T4 构建、正确性、回归和正式 vLLM benchmark 已完成。"
echo "结果目录：${RESULT_DIR}"

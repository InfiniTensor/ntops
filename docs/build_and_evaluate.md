# 一键构建与评测

## 1. 环境要求

- Linux、Python 3.10 或更高版本；
- 已适配目标设备的 PyTorch，以及可用的 CUDA、HIP 或 CoreX runtime；
- `ninetoothed>=0.16.0`；
- 海光 DCU 或天数智芯 MR-V100。仅执行构建时不要求加速卡。

## 2. 一键执行

构建、定向正确性测试和四个算子 benchmark 使用同一个入口：

```bash
./scripts/build_and_evaluate.sh all
```

脚本默认执行 editable install，打印 Python、PyTorch、CUDA/HIP、NineToothed 实际导入路径和设备信息，然后运行测试及 benchmark。日志写入 UTC 时间戳目录：

```text
artifacts/YYYYmmddTHHMMSSZ/
  build.log
  environment.txt
  pytest.log
  bench_*.csv
```

### 2.1 Build 实际命令

脚本的 build 步骤执行：

```bash
python -m pip install -e ".[testing]"
```

这一步安装当前源码及测试依赖，不执行 NineToothed kernel 的 AOT 编译。kernel 在测试或 benchmark 第一次调用对应算子时 JIT 编译；启用多个候选时，自动调优也在该阶段完成。`build` 模式在安装并记录环境信息后退出：

```bash
./scripts/build_and_evaluate.sh build
```

可按阶段运行：

```bash
./scripts/build_and_evaluate.sh build
./scripts/build_and_evaluate.sh test
./scripts/build_and_evaluate.sh benchmark
```

已经安装依赖时可跳过安装；也可指定解释器和结果目录：

```bash
SKIP_INSTALL=1 PYTHON_BIN=/path/to/python \
RESULT_ROOT=/path/to/results ./scripts/build_and_evaluate.sh all
```

## 3. 手工等价命令

```bash
python -m pip install -e ".[testing]"
PYTHONPATH=src python -m pytest -q \
  tests/test_block_scaled_fp8_mm.py \
  tests/test_mxfp4_w4a16_grouped_mm.py \
  tests/test_rms_norm_gated.py \
  tests/test_fused_mla_rope_cache_write.py
```

```bash
PYTHONPATH=src python benchmarks/bench_block_scaled_fp8_mm.py
PYTHONPATH=src python benchmarks/bench_mxfp4_w4a16_grouped_mm.py
PYTHONPATH=src python benchmarks/bench_rms_norm_gated.py
PYTHONPATH=src python benchmarks/bench_fused_mla_rope_cache_write.py
```

benchmark 使用 `triton.testing.do_bench(warmup=25, rep=100, return_mode="mean")`，结果单位为微秒。首次编译与自动调优应在正式计时前完成。量化算子的 CSV 中必须同时保留 `reference_provider` 和输入存储类型；软件反量化 reference 的加速比不能解释为相对平台原生量化算子的收益。

## 4. 两平台复验

海光和天数智芯必须分别在同一提交上执行 `all`，不得共用另一平台的调优配置或性能结论。归档时保留：

- `git rev-parse HEAD` 与工作区是否干净；
- 脚本生成的环境、测试和 benchmark 日志；
- 设备型号、驱动/runtime 版本；
- 量化 benchmark 实际采用的 reference provider；
- 自动调优选出的 `num_warps`、`num_stages`。

HIP 原生编译器错误可能结束整个 Python 进程。引入新的 tile、dot 或多 wave 候选前，应在独立进程逐个编译和校验，不能直接加入进程内自动调优集合。

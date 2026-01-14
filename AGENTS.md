# Repository Guidelines

## 项目结构与模块组织
- `src/ntops/` 是包入口；`kernels/` 放 NineToothed kernel 定义（每个算子一个文件），`torch/` 是 PyTorch 侧封装与调用。
- `tests/` 使用 pytest，单算子测试如 `tests/test_add.py`，共享夹具在 `tests/conftest.py` 与 `tests/utils.py`。
- 根目录 `README.md` 为项目概览，`doc.md`/`topk.md` 记录算子笔记与设计说明。

## NineToothed 依赖与接口要点
- NineToothed 是基于 Triton 的 DSL，核心范式为 TOM：`arrangement`（布局变换）+ `application`（算子表达）+ `tensors`（符号张量定义）。
- `ninetoothed.Tensor` 表达符号张量，`tile/expand/permute` 等用于构建层级布局；`ninetoothed.block_size()` 提供可调块大小符号。
- `ninetoothed.language as ntl` 提供算子接口（如 `ntl.zeros`、`ntl.dot`）；`ninetoothed.make(...)` 负责 JIT 生成可调用 kernel。
- 在 ntops 中，每个 kernel 提供 `premake(...) -> (arrangement, application, tensors)`；`src/ntops/torch/utils.py` 中 `_cached_make` 统一调用 `ninetoothed.make`。

## 构建、测试与开发命令
- `python -m pip install -e .` 以 editable 方式安装；`python -m pip install -e ".[testing]"` 安装测试依赖。
- `python -m pytest` 运行全量测试；`python -m pytest tests/test_mm.py` 运行单文件。
- `python -m ruff check .` 运行 lint（ruff）。

## 编码风格与命名约定
- Python 4 空格缩进，snake_case；算子文件与入口函数保持同名（如 `kernels/gelu.py`）。
- `application` 中仅写符号计算，避免引入真实张量运算或框架 side-effect。
- `pyproject.toml` 已启用 ruff 的错误/导入规则，保持导入顺序一致。

## 测试指南
- 使用 pytest；新增算子测试命名 `test_<op>.py`，优先使用 `pytest.mark.parametrize`。
- 精度容差与已有测试一致（通常为 `rtol, atol`），必要时在测试内说明原因。

## 提交与 PR 规范
- 提交信息简洁、动词开头（示例："Add matmul operator"、"Refactor tests"）。
- PR 需描述改动、附测试结果，并关联相关 issue 或设计文档（如 `doc.md`）。

# REFERENCE.md — T1-1-4 赛题参考资源声明

本次赛题提交属于**情况 2**：有参考外部开源代码及相关资源。以下按比赛要求逐项列出每个参考资源的信息。

---

## 参考 1：PyTorch 开源项目

1. **参考开源项目/资源名称**：PyTorch

2. **参考资源链接**：https://github.com/pytorch/pytorch

3. **参考的具体内容**：
   - `torch.roll`、`torch.column_stack`、`torch.mode`、`torch.meshgrid`、`torch.cartesian_prod` 的 Python API 签名和语义规范（参数名称、默认值、返回值形状、dtype 规则、错误处理行为）。
   - `torch.mode` 的 CUDA kernel 源码（`aten/src/ATen/native/cuda/SummaryOps.cu`），确认 PyTorch 2.12 CUDA 实现采用排序 + BlockReduce 的并行拓扑，平局行为由并行归约树决定，官方文档明确标注为"undefined (but deterministic)"。九齿 kernel 采用确定性策略（最小值 + 最大下标），测试改为语义验证（验证返回值频次等于最大频次）。
   - `torch.column_stack` 的 C++ 实现（`aten/src/ATen/native/TensorShape.cpp`），确认 0D/1D 输入先 reshape 为 `(numel, 1)` 再沿 dim=1 拼接的语义。
   - `torch.meshgrid` 接受 0D 张量（视为长度 1）和单 list/tuple 参数的接口约定。

4. **本人对参考内容的修改与优化说明**：
   - 所有算子的数学语义（roll 的环形平移、meshgrid/cartesian_prod 的索引映射、column_stack 的拼接逻辑、mode 的频次统计）通过输出驱动 gather 模式在九齿 application 函数内重新实现，不调用 PyTorch 参考算子完成核心计算。
   - roll 采用 permute + ntl.load 手工指针寻址，支持 1D-5D；浮点取模用 `(offsets - shift + size) % size` 避免 Triton 负数余数问题。
   - column_stack 的核心创新是利用九齿 arrangement 的 stride-aware `flatten().tile()` 直接写入非连续输出切片，无需临时缓冲区。
   - meshgrid 复用了 cartesian_prod 的整除取模 kernel，两者共享 `output_k[flat] = input_k[(flat // repeat_after[k]) % Sk]` 的索引公式。
   - mode 通过 O(K²) 逐行统计算法 + K_orig/K_tile mask 方案在应用层实现了规约，绕过了 arrangement 层 expand 的 0.25.0 bug。

5. **开源协议类型**：PyTorch 使用 BSD-3-Clause License。

6. **其他补充说明**：无。

---

## 参考 2：Ninetoothed（九齿）框架

1. **参考开源项目/资源名称**：ninetoothed（九齿 GPU 算子代码生成框架）

2. **参考资源链接**：https://github.com/InfiniTensor/ninetoothed

3. **参考的具体内容**：
   - `ninetoothed/generation.py`：code generator 的 `visit_Call` 方法（第 206-262 行）和 `visit_Subscript` 方法（第 268-294 行），确认 `tensor.data_ptr()`、`tensor.offsets(dim)`、`tensor.stride(dim)`、`tensor.source.shape[dim]` 在 application 函数中的可用性和约束。确认 `stride(dim)` 要求 dim 为正整数字面量（`stride(-1)` 存在代码生成 bug）。
   - `ninetoothed/tensor.py`：Tensor 类的 `tile`、`expand`、`flatten`、`permute`、`unsqueeze`、`squeeze`、`__getitem__` 方法在 arrangement 层的语义和索引映射实现。确认 arrangement 层无 split/concat/roll/scatter 等数据重排布原语。
   - `ninetoothed/symbol.py`：Symbol 类的算术运算符重载（`__add__`、`__sub__`、`__mul__`、`__floordiv__`、`__mod__`），确认 offsets 表达式可参与算术运算。
   - `ninetoothed/language.py`：`ntl.*` 操作通过 AST `call` 和 `attribute` 函数映射到 `triton.language.*` 的机制。
   - 现有算子参考：`src/ntops/kernels/element_wise.py` 的 `arrangement`（`flatten().tile()` 模式）和 `src/ntops/kernels/silu.py` 的 premake 结构（`functools.partial` + `Tensor` 构造 + `_cached_make` 调用链）。

4. **本人对参考内容的修改与优化说明**：
   - 系统性地探针了九齿 0.25.0 的能力边界，发现 `ntl.load(ptr)` 手工指针路径可用（核心突破）、`input.source[dynamic_idx]` 下标存在 IndexError bug、`arrangement` 层的 `unsqueeze + expand` 只广播第一个元素（bug）、`break` 语句不支持等限制。
   - 针对 0.25.0 的限制，所有数据重排布算子采用 output-driven gather 模式，将索引映射逻辑从 arrangement 层下沉到 application 层的手工指针算术中。
   - mode 算子通过 `ntl.where` + `ntl.sum` + `ntl.cast` 的组合克服了双重循环变量类型不匹配的问题，通过 `valid = ntl.cast(offsets(1) < K_orig, ntl.int32)` mask 排除 padding lane。
   - 阅读框架源码以确认 API 边界，属于对开发工具的正常理解，不涉及对九齿框架本身的修改。

5. **开源协议类型**：随 ntops 仓库分发，ntops 使用 Apache 2.0 License。

6. **其他补充说明**：九齿是本赛题指定的开发框架，使用其为正常参赛行为。

---

## 参考 3：Triton 语言

1. **参考开源项目/资源名称**：Triton

2. **参考资源链接**：https://github.com/triton-lang/triton

3. **参考的具体内容**：
   - `triton.language` 的 API：`load`、`sum`、`where`、`cast`、`arange`、`program_id` 等原语的语义和参数约定。
   - 确认 `tl.load` 可以接受手工构造的指针（非仅 arrangement 生成的指针），支持 scalar pointer + manual offset 计算。
   - 确认 `tl.sum` 可以规约向量到标量，支持在 application 函数中实现跨维度的计数统计。
   - 确认 `tl.arange` 要求 range 大小为 2 的幂（导致 mode 的 K 需要补齐到 next_power_of_2）。

4. **本人对参考内容的修改与优化说明**：
   - 九齿 DSL 层不直接暴露所有 Triton 原语（如 `load`、`sum`、`where`、`cast`），通过阅读 `generation.py` 确认 code generator 会将 `ninetoothed.language.X` 无条件映射为 `triton.language.X`，从而在 application 函数中直接使用这些原语。
   - 发现了 Triton `%` 对负数产生负余数的问题，roll kernel 采用 `(offsets - shift + size) % size` 保证结果非负。
   - 这是对开发工具链的正常理解，不涉及对 Triton 本身的修改。

5. **开源协议类型**：Triton 使用 MIT License。

6. **其他补充说明**：Triton 是九齿的底层编译后端，使用其为正常参赛行为。

---

## 参考 4：Codex（OpenAI）开发辅助

1. **参考资源名称**：Codex（OpenAI 编程助手）

2. **参考资源链接**：通过 Claude Code IDE 集成访问

3. **参考的具体内容**：
   - 提供了架构设计建议：确认 output-driven gather 模式在九齿中可行，建议使用 `ntl.load` 手工指针替代 `source[idx]` 下标（因 0.25.0 bug）。
   - 提供了 mode 的实现方向：O(K²) 逐行统计算法、K_orig + K_tile + mask 方案、`ntl.sum` + `ntl.where` 的用法。
   - 审查了代码质量和 API 边界覆盖（0D 输入、单 list 参数、非 1D 报错等），指出了参数校验和测试覆盖的遗漏。
   - 审查了 REFERENCE.md 和 HONOR_CODE.md 的格式和完整性。

4. **本人对参考内容的修改与优化说明**：
   - 所有最终代码由本人编写、调试和验证。Codex 提供的是方向性建议，具体实现（如 ndim 特化的 application 函数、安全取模公式、mask 的 `* valid` 而非 `& (valid > 0)` 的 Triton 语义修正）均为本人独立完成。
   - 九齿 0.25.0 的能力边界探针由本人编写和验证，Codex 未参与探针的代码编写。

5. **开源协议类型**：不适用（商业服务）。

6. **其他补充说明**：Codex 作为 AI 编程助手提供代码审查和架构建议，属于参赛规则允许的工具使用范围。

---

## 补充声明

除上述明确标注的参考资源外，本次赛题提交的算子代码的核心算法逻辑及实现方案均为本人在比赛期间独立设计与开发。所有代码均使用九齿 DSL（arrangement/application/premake）或 Triton 语言从零编写，未直接复制任何外部项目的代码片段。

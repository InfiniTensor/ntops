# REFERENCE.md — T1-1-9 赛题参考资源声明

本次赛题提交属于**情况2**：有参考外部开源代码及相关资源。
以下按比赛要求逐项列出每个参考资源的信息。

---

## 参考 1：PyTorch 开源项目

1. **参考开源项目/资源名称**：PyTorch

2. **参考资源链接**：https://github.com/pytorch/pytorch

3. **参考的具体内容**：
   - `aten/src/ATen/native/FractionalMaxPool2d.cpp`：fractional_max_pool2d 的 CUDA 实现，用于确认池化窗口起点计算公式（`alpha = (input_size - kernel_size) / (output_size - 1)`，`start[i] = int((i + sample) * alpha) - int(sample * alpha)` 等）、`_random_samples` 的维度和语义（shape `(N, C, 2)`，index 0 = W sample，index 1 = H sample）、最大值选择逻辑（`val > maxVal || isnan(val)`）、返回 indices 的格式（空间维扁平索引，不含 N/C）。
   - `aten/src/ATen/native/FractionalMaxPool3d.cpp`：3D 扩展，sample 顺序（D, H, W）、flat spatial index 计算。
   - `torch/nn/functional.py`：`multilabel_margin_loss` 的函数签名与语义。
   - `torch/nn/modules/pooling.py`：`FractionalMaxPool2d`/`FractionalMaxPool3d` 参数约定。

4. **本人对参考内容的修改与优化说明**：
   - 池化窗口公式的核心数学逻辑（alpha 计算、起点公式、last-window 边界处理）被精确复现以保证与 PyTorch 的输出一致性。并非直接复制 PyTorch C++ 代码，而是在九齿（ninetoothed）框架内用其 DSL（`ntl.where`、`ntl.cast`、`ntl.load`、pointer 算术等）重新实现了完整的 output-driven 2D/3D kernel。
   - PyTorch 代码中 pool_start 在 CPU 侧预计算；本实现将公式完整下沉到 GPU kernel（application 函数）内，通过 `output.offsets(dim)` 解码多维坐标后实时计算。
   - 增加了 tail mask（`valid = (n<N)&(c<C)&(oh<Ho)&(ow<Wo)`）保护 padding lane 越界，PyTorch 使用隐式边界检查。
   - multilabel_margin_loss 的公式通过实验反推后，在九齿内用 O(C²) offsets-based 设计实现，非直接翻译 PyTorch C++ 代码。

5. **开源协议类型**：PyTorch 使用 BSD-3-Clause License。

6. **其他补充说明**：无。

---

## 参考 2：Ninetoothed（九齿）框架

1. **参考开源项目/资源名称**：ninetoothed（九齿 GPU 算子代码生成框架）

2. **参考资源链接**：https://github.com/InfiniTensor/ntops（随 ntops 仓库安装为 `pip install -e .` 的依赖）

3. **参考的具体内容**：
   - `ninetoothed/generation.py`：code generator 的 `visit_Call` 方法（第 206-262 行），确认 `tensor.data_ptr()`、`tensor.offsets(dim)`、`tensor.stride(dim)` 在 application 函数中的可用性和约束。
   - `ninetoothed/tensor.py`：Tensor 类的 `tile`、`flatten`、`ravel`、`offsets` 等方法的语义。
   - `ninetoothed/symbol.py`：Symbol 算术运算符不继承 bounds 的行为（`__add__`、`__mul__` 等创建裸 AST 节点）。
   - `ninetoothed/language.py`：`ntl.*` 操作通过 AST 映射到 `triton.language.*` 的机制。

4. **本人对参考内容的修改与优化说明**：
   - 阅读框架源码以确认 API 边界，属于对开发工具的正常理解，不涉及对九齿框架本身的修改。
   - 发现了 autotuning warmup 对有副作用 kernel（atomic_add）的数据污染问题，通过固定 `block_size` 为 Python int 绕开。
   - 发现了全 concrete shape（`Tensor(shape=(N, C, H, W))`）可消除 Symbol 算术对 autotuning 的影响。

5. **开源协议类型**：随 ntops 仓库分发，ntops 使用 Apache 2.0 License。

6. **其他补充说明**：九齿是本赛题指定的开发框架，使用其为正常参赛行为。

---

## 参考 3：Triton 语言

1. **参考开源项目/资源名称**：Triton

2. **参考资源链接**：https://github.com/triton-lang/triton

3. **参考的具体内容**：
   - `triton.language` 的 API 文档：`atomic_add`、`load`、`store`、`where`、`maximum`、`sum`、`cast`、`floor`、`ceil` 等原语的语义和参数。
   - 确认 `tl.atomic_add` 支持向量化调用。
   - 确认 `tl.trunc` 在当前版本不可用，改用 `where/floor/ceil` 实现。

4. **本人对参考内容的修改与优化说明**：
   - 九齿 DSL 层不直接暴露所有 Triton 原语（如 `atomic_add`、`load`），通过阅读 generation.py 确认 code generator 会将 `ninetoothed.language.X` 无条件映射为 `triton.language.X`，从而在 application 函数中直接使用这些原语。
   - 这是对开发工具链的正常理解，不涉及对 Triton 本身的修改。

5. **开源协议类型**：Triton 使用 MIT License。

6. **其他补充说明**：Triton 是九齿的底层编译后端，使用其为正常参赛行为。

---

## 参考 4：Fractional MaxPooling 论文

1. **参考资源名称**：Fractional Max-Pooling

2. **参考资源链接**：https://arxiv.org/abs/1412.6071

3. **参考的具体内容**：Benjamin Graham 提出的 fractional max pooling 算法的原始描述，包括随机步长池化的概念和基本公式。用于理解算子背景，不涉及代码参考。

4. **本人对参考内容的修改与优化说明**：
   - 仅参考算法概念。实际公式以 PyTorch 2.12 CUDA 实现为准（见参考 1），论文公式与 PyTorch 实现存在差异。
   - 九齿 kernel 的窗口起点计算、窗口扫描逻辑、indices 计算均为独立开发。

5. **开源协议类型**：学术论文，非软件项目。

6. **其他补充说明**：无。

---

## 补充声明

除上述明确标注的参考资源外，本次赛题提交的算子代码的核心算法逻辑及实现方案均为本人在比赛期间独立设计与开发。所有代码均使用九齿 DSL（arrangement/application/premake）或 Triton 语言从零编写，未直接复制任何外部项目的代码片段。

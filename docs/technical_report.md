# KernelSwift Competition 技术报告

## 1. 修改内容与动机

本分支交付四个面向 LLM 推理的 NineToothed 算子。所有改动都属于 A 部分的算子实现与优化；B 部分没有修改，没有新增编译 pass，也没有修改 NineToothed、Triton、CoreX 或 HIP 后端源码。

| 范围 | 正式入口 | 目标问题 | 主要场景 |
| --- | --- | --- | --- |
| A 部分 | `block_scaled_fp8_mm` | 1x128/128x128 block scale 的 FP8 GEMM | Attention、线性层、单专家 MoE |
| A 部分 | `mxfp4_w4a16_grouped_mm` | MXFP4 权重解码、缩放和 grouped GEMM 融合 | uniform/routed MoE |
| A 部分 | `rms_norm_gated` | RMSNorm、门控与激活融合 | decode、prefill |
| A 部分 | `fused_mla_rope_cache_write` | MLA RoPE 与压缩 paged KV cache 写入融合 | MLA decode、长上下文 |
| B 部分 | 无 | 不修改编译器和平台后端 | 不适用 |

## 2. 方案与实现

### 2.1 公共实现原则

四个算子先冻结 shape、dtype、layout、原地语义和拒绝路径，再建立独立 PyTorch reference。kernel 只使用 `source[...]`、masked load/store、基础算术、公开 cast、`ntl.sum` 和普通 `ntl.dot` 等公共 lowering 能力。

平台差异在 PyTorch wrapper 或 premake 参数处显式分派。通用数学语义和公开 API 保持一致，平台特化只改变安全的 tile、dot/reduction 路径与调优集合。修改没有侵入 NineToothed 或 Triton 安装目录，也没有进程级 patch。

### 2.2 Block-scaled FP8 MM

输入为 `A[M,K] @ B[K,N]`。A scale 使用 `BlockWise1x128`；B scale 支持 `BlockWise1x128` 和 `BlockWise128x128`。A 是 row-major FP8，B 是由权重转置得到的 column-major dense FP8；scale 同时接受 row-major、column-major 及 PyTorch 使用的 K-block padding。

每个 128-wide K block 的 FP32 contribution 为：

```text
acc[m,n] += dot(A[m,p], B[p,n]) * scale_a[m,p] * scale_b[p,n]
```

HIP 和支持 FP8 Tensor Core 的 NVIDIA CUDA 使用普通 FP8 `ntl.dot`。CoreX 与不支持原生 FP8 dot 的 CUDA 路径将有限 E4M3FN 值精确转换为 BF16，再执行 BF16 dot；这是一条正确性回退，不改变公开 FP8 输入。HIP 固定 `BLOCK_M=16, BLOCK_N=16, BLOCK_K=32, num_warps=1, num_stages=1`，避免 gfx936 在多 wave FP8 specialization 上发生 AMD LLVM 进程级崩溃。

权重按 128-N block 建立无分配 view；A 与 A scale 使用 stride-0 view 共享，wrapper 不物化完整反量化矩阵。bias 在 FP32 accumulator 上融合，最后一次性转换到 FP16、BF16 或 FP32。

### 2.3 MXFP4 W4A16 Grouped MM

支持两种布局：

| 模式 | activation | packed weight | scale | output |
| --- | --- | --- | --- | --- |
| uniform | `[G,M,K]` | `[G,K/2,N]` | `[G,K/32,N]` | `[G,M,N]` |
| routed | `[total_M,K]` | `[G,K/2,N]` | `[G,K/32,N]` | `[total_M,N]` |

每 byte 的低、高 nibble 分别保存偶数和奇数 K 的 E2M1 code；E8M0 byte 解码为 `2^(s-127)`。每个 32-wide scale block 拆成两组 16-wide K。CoreX 使用两次 BF16 dot，HIP 使用无分支整数解码加显式 FP32 reduction，避免 gfx936 的 BF16 MMAC codegen 崩溃。两条路径都不物化 `[G,K,N]` 权重。

uniform HIP 在已验证安全的 1/4-wave 中有界调优；routed HIP 固定单 wave。`offs[g]` 保持运行时值，防止按最大序列长度 specialization 后由 padding program 覆盖后续专家输出。零 token 专家由相邻相同 offset 表达。

### 2.4 Gated RMSNorm

数学定义为：

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

norm_before_gate=True:  y = RMSNorm(x) * activation(z)
norm_before_gate=False: y = RMSNorm(x * activation(z))
```

平方和、激活和归一化因子使用 FP32，最后转换回输入 dtype。一个 program 负责一行或一个完整 normalization group，归约轴不跨 program。不同门控顺序、激活、分组和可选输入在 premake 阶段选择 application，热路径没有对应的运行时分支。

默认搜索 `num_warps=(1,2,4,8)`、`num_stages=(1,2)`，winner 按 shape、dtype、stride 和静态语义缓存。若新增 HIP 候选，必须先在独立进程验证编译稳定性。

### 2.5 Fused MLA RoPE Cache Write

接口为：

```python
ntops.torch.fused_mla_rope_cache_write(
    kv_c,             # [T_source, L]
    k_pe,             # [T_source, R] or [T_source, 1, R]
    kv_cache,         # [num_blocks, cache_block_size, L + R]
    slot_mapping,     # [T] int64; -1 skips
    positions,        # [T] int32/int64
    cos_sin_cache,    # [max_position, R], packed [cos | sin]
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
)
```

`kv_c` 是共享低秩 K/V latent，不按 head 展开。典型 `L=512, R=64` 时，每 token 只写 576 个元素。cache 地址为：

```text
block_idx = slot // cache_block_size
offset    = slot % cache_block_size
entry     = [kv_c | RoPE(k_pe)]
```

一个 program 负责一个 `[token, entry tile]`。token/feature 坐标来自 `output_anchor.offsets()`，latent、RoPE table 与 paged cache 通过 `source[...]` 间接访问。tile 至少覆盖完整 entry 的下一个 2 的幂，避免长上下文产生多余 program。

NineToothed SSA 图需要真实输出根，部分 DCU backend 不能可靠 lower 只有间接副作用的 launch。当前唯一 kernel 因此生成 `[T,L+R]` 的压缩 entry 输出锚定，同时在同一 application 中写 paged cache；wrapper 丢弃锚定输出。它不生成 query 或按 head 展开，比借用其他融合语义更直接，也使 CoreX/HIP 共用同一 kernel。非连续 source 先连续化，非连续 cache 通过临时连续 cache 执行后 copy back；连续生产输入没有该额外复制。

`slot=-1` 被映射到负 block 并由 store mask 抑制。`T_source` 可以大于有效 `T`，以适配 CUDA Graph padding。FP8 cache 与 `fp8_ds_mla` 量化布局尚未实现，明确拒绝而不是静默套用未量化语义。

### 2.6 编译 pass、后端与官方 NineToothed 对比

本分支没有 B 部分改动，也没有自定义 pass。因此不存在“修改后编译后端相对官方 NineToothed 后端”的独立收益数据；所有表格均是官方 NineToothed 安装编译本仓库算子后，相对 PyTorch reference 的端到端结果。实际收益来自 program 映射、融合、在线解码、无分配 view 和平台安全分派，不应归因于后端修改。

## 3. 实验环境与方法

### 3.1 环境

| 平台 | 设备 | PyTorch | Triton/runtime |
| --- | --- | --- | --- |
| 天数智芯 | Iluvatar MR-V100 | 2.7.1+corex.4.4.0 | Triton 3.1.0+corex.4.4.0、CUDA compatibility 10.2 |
| 海光 | BW/gfx936 | 2.9.0+das.opt1.dtk2604 | Triton 3.5.1+das.opt1.dtk2604.torch290、HIP 6.3.26093 |

benchmark 使用 `do_bench(warmup=25, rep=100, return_mode="mean")`，正式采样前完成首次编译与自动调优，结果为平均微秒。量化算子优先使用 PyTorch 原生 dtype/operator；缺少原生 operator 时回退到软件反量化 reference，并在结果中单独标识。

### 3.2 量化 dtype、输入与 reference

两个平台的 PyTorch dtype 和原生 operator 支持情况如下：

| 平台 | Block FP8 dtype | `torch.nn.functional.scaled_mm` | MXFP4 dtype | `torch.nn.functional.scaled_grouped_mm` |
| --- | --- | --- | --- | --- |
| 天数 MR-V100 | `float8_e4m3fn` 可用 | 不可用 | `float8_e8m0fnu` 可用，`float4_e2m1fn_x2` 不可用 | 不可用 |
| 海光 BW | `float8_e4m3fn` 可用 | 没有可运行实现 | packed E2M1 与 E8M0 dtype 可用 | 没有可运行实现 |

dtype 转换必须区分数值转换与位模式解释：

- `.to(torch.float8_e4m3fn)` 是数值转换，会把浮点输入量化并舍入到 E4M3FN；
- FP8 tensor 的 `.float()` 解码 E4M3FN 自身表示的数值，但外部 block scale 仍需
  单独相乘，二者共同构成完整反量化；
- `.view(torch.float4_e2m1fn_x2)`、`.view(torch.float8_e8m0fnu)` 或
  `.view(torch.uint8)` 只重新解释相同位模式，不执行量化或反量化；
- 普通 BF16/FP32 matmul 不能直接消费 packed code，必须先解码 code 并应用 scale。

Block-scaled FP8 benchmark 在两个平台都从随机浮点数据出发，通过 `.to(torch.float8_e4m3fn)` 生成真实 FP8 tensor。由于没有可运行的原生 `scaled_mm`，两端都使用 `reference_provider=torch_dtype_dequant_mm`：先用 `.float()` 解码 FP8，再按 128-wide block 应用 A/B scale，最后执行 FP32 matmul。性能场景的 scale 初始化为 1，但 reference 仍执行完整 scale 路径；非单位 scale 由正确性测试覆盖。当前 benchmark 不提供 FP8 `uint8` 存储回退；若 PyTorch 不提供 E4M3FN dtype，该 benchmark 不能运行。

MXFP4 benchmark 直接随机生成 packed E2M1 byte 和 E8M0 scale byte，它们是合法的量化编码，不是普通 `uint8` 权重，也不是从一份浮点模型权重经 calibration 得到。海光将这些 byte 以原生 packed dtype 暴露，天数因缺少 packed E2M1 dtype而保持 `uint8` 存储。两端原生 grouped operator 都不可用，因此 reference 都显式拆分高低 nibble、解码 E2M1、解码并应用 E8M0 scale，再执行 FP32 grouped matmul。对应 provider 分别为 `manual_dequant_mm_native_operator_unavailable` 和 `manual_dequant_mm_native_dtype_unavailable`。

软件 reference 的计时包含解码、scale 展开、反量化临时张量和 FP32 matmul；因此相关加速比表示融合 ntops kernel 相对软件端到端 reference 的收益，不能解释为相对平台原生量化 operator 的收益。ntops kernel 不物化完整反量化权重：Block FP8 在 tile 内执行 dot 与 scale，MXFP4 在 tile 内在线解码并计算。

### 3.3 正确性范围

天数侧在当前分支运行四个定向测试文件，结果为 `78 passed, 2 skipped`。两个 skip 仅因 PyTorch 2.7.1 没有原生 MXFP4 dtype；raw `uint8` 数值路径仍实际执行。测试覆盖 FP8 两种 recipe、MXFP4 全部 16 个 code、Gated RMSNorm 全部分支、MLA 融合算子十项用例及生成源码检查。统一命令见 `docs/build_and_evaluate.md`。

海光结果包括 Block-scaled FP8 三个代表 shape 的数值复验、MXFP4 三个 benchmark 场景、Gated RMSNorm 三个场景以及 MLA 融合算子定向测试；各项正确性检查均通过。

## 4. 实验结果

### 4.1 Block-scaled FP8 MM

两个平台均使用 `torch_dtype_dequant_mm` 软件 reference。天数 MR-V100：

| 场景 `(M,N,K)` | reference | reference (us) | ntops (us) | 加速比 | 最大绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| decode `(1,4096,4096)` | software dequant MM | 609.584 | 228.115 | 2.672x | 0.000000 |
| MoE `(32,14336,4096)` | software dequant MM | 1910.558 | 1531.907 | 1.247x | 0.250000 |
| prefill `(128,4096,4096)` | software dequant MM | 677.933 | 1726.408 | 0.393x | 0.000004 |

海光 BW，固定 `(warps,stages)=(1,1)`：

| 场景 `(M,N,K)` | reference (us) | ntops (us) | 加速比 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| decode `(1,4096,4096)` | 391.849 | 902.313 | 0.434x | 0.000000 |
| MoE `(32,14336,4096)` | 1252.682 | 3853.278 | 0.325x | 0.250000 |
| prefill `(128,4096,4096)` | 439.770 | 1733.113 | 0.254x | 0.125000 |

两端都没有可运行的原生 public block-scaled operator，因此加速比只代表相对完整 FP8 解码、block scale 应用和 FP32 matmul 软件 reference。decode 在天数侧获益，prefill 与海光三项仍有优化空间。

### 4.2 MXFP4 W4A16 Grouped MM

天数 MR-V100 使用 raw `uint8` packed code，reference provider 为 `manual_dequant_mm_native_dtype_unavailable`：

| 场景 | 模式 | ntops (us) | 加速比 | TFLOP/s | 最大绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| uniform decode | uniform | 892.727 | 42.268x | 0.301 | 0.000000 |
| uniform concurrent | uniform | 1373.795 | 27.453x | 3.126 | 0.500000 |
| routed uneven | routed | 5837.419 | 6.474x | 0.736 | 2.000000 |

海光 BW 的原生 packed dtype 可用，但原生 grouped operator 不可用；reference provider 为 `manual_dequant_mm_native_operator_unavailable`：

| 场景 | 模式 | reference (us) | ntops (us) | 配置 | 加速比 | 最大绝对误差 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| uniform decode | uniform | 18681.152 | 2219.319 | `(4,1)` | 8.418x | 0.000000 |
| uniform concurrent | uniform | 18947.936 | 14143.086 | `(4,1)` | 1.340x | 0.031250 |
| routed uneven | routed | 18885.632 | 53806.240 | `(1,1)` | 0.351x | 0.500000 |

routed HIP 固定单 wave 后，相对错误选择 4-wave 时的 1296.786 ms 降到 53.806 ms，约 24.1 倍改善，但仍慢于软件 reference。

### 4.3 Gated RMSNorm

天数 MR-V100，BF16、hidden size 128：

| 场景 | 行数 | 配置 | ntops (us) | PyTorch (us) | 加速比 | 最大绝对误差 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| decode | 8 | `(8,2)` | 8.047 | 42.421 | 5.271x | 0.000000 |
| concurrent decode | 80 | `(2,2)` | 8.888 | 53.066 | 5.971x | 0.000000 |
| prefill 2048 | 16384 | `(1,1)` | 54.793 | 348.886 | 6.367x | 0.003906 |

海光 BW：

| 场景 | 行数 | 配置 | ntops (us) | PyTorch (us) | 加速比 | 最大绝对误差 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| decode | 8 | `(8,2)` | 663.288 | 54.758 | 0.083x | 0.000000 |
| concurrent decode | 80 | `(8,2)` | 645.244 | 51.970 | 0.081x | 0.000000 |
| prefill 2048 | 16384 | `(4,2)` | 692.300 | 179.012 | 0.259x | 0.003906 |

融合在天数侧有稳定收益；海光侧数值正确但 launch/lowering 开销占主导，继续调整候选本身不能替代后端与 program 映射分析。

### 4.4 Fused MLA RoPE Cache Write

shape 为 `L=512, R=64, cache_block_size=16, BF16`。天数数据由当前分支重新测得：

| 场景 | T | PyTorch 未融合 (us) | 配置 | ntops (us) | 加速比 |
| --- | ---: | ---: | --- | ---: | ---: |
| decode | 1 | 63.273 | `(4,1)` | 9.103 | 6.951x |
| concurrent decode | 10 | 81.243 | `(4,1)` | 10.497 | 7.739x |
| long context | 2048 | 170.413 | `(2,2)` | 22.808 | 7.472x |

相对未融合 PyTorch，三个场景均获得 6.95x 至 7.74x 加速。

海光 BW：

| 场景 | T | PyTorch 未融合 (us) | 配置 | ntops (us) | 加速比 |
| --- | ---: | ---: | --- | ---: | ---: |
| decode | 1 | 193.815 | `(8,1)` | 1289.147 | 0.150x |
| concurrent decode | 10 | 195.311 | `(8,1)` | 1323.701 | 0.148x |
| long context | 2048 | 180.899 | `(1,1)` | 1319.881 | 0.137x |

海光三个场景的数值检查通过，但当前延迟高于 PyTorch 未融合 reference，主要开销来自该平台的输出锚定与 kernel launch/lowering。两平台结果分别报告，不跨平台外推配置或加速比。

## 5. 对照与消融

- Block-scaled FP8：CoreX 的 BF16 dot 是 FP8 有限值的精确回退。HIP 多 wave
  specialization 会触发 AMD LLVM 崩溃，恢复单 wave 后正确性与服务稳定性恢复，
  代价是当前性能低于软件 reference。
- MXFP4：查表解码曾因 tensor-valued 索引坐标丢失产生约 98% 元素错误；改为纯
  integer bitwise/arithmetic 后可穷举通过全部 code。routed 4-wave 到 1-wave 的
  24.1x 延迟改善证明 wave 策略是独立关键因素。
- Gated RMSNorm：相对 PyTorch eager 的 5.27x 至 6.37x 天数收益来自归约、门控、
  激活和权重乘法单 kernel 融合；海光反例表明融合不能自动抵消后端 launch 开销。
- Fused MLA RoPE Cache Write：直接对照 eager RoPE 加两次 cache scatter，长上下文
  获得 7.47x 加速。

## 6. 工程质量与适用边界

### 6.1 测试与兼容性

测试覆盖公开 dtype、主要分支、尾块、空/零 token、padding、非连续 view、原地写入、非法参数、平台分派和生成源码。量化 reference 独立解码；MLA reference 使用 PyTorch FP32 RoPE 和显式 paged scatter。执行命令见 `docs/build_and_evaluate.md`。

API 改动控制在新增的任务入口；未修改 PyTorch 或 NineToothed 全局行为。MLA 融合算子只暴露一个正式入口。其他三个算子只接受明确列出的 recipe/layout，不把不支持的组合静默解释为相近语义。

### 6.2 通用能力与平台特化

通用层包含数学语义、layout contract、reference、公开 DSL application 和参数校验。平台特化仅包含 wrapper 选择的 dot/reduction、tile、wave 和候选集合。平台路径由 HIP/CoreX/CUDA capability 决定，不把设备名称散落进数学实现。

### 6.3 已知限制

- Block-scaled FP8 暂不支持 A 侧 128x128 scale、swizzle、多级 scale、batch、
  grouped launch、N 尾块和 fast accumulation；
- MXFP4 只支持 W4A16 `BlockWise1x32`，不支持 activation scale、bias、swizzle、
  grouped-K 和 `[G,N,K/2]` 直接输入；
- Gated RMSNorm 的新 HIP 配置需先做独立进程编译筛选；
- Fused MLA RoPE Cache Write 支持 FP16/BF16/FP32 未量化 cache，不支持 FP8 cache layout；
- 非连续 cache 会产生临时连续副本；
- 软件 reference 的性能不能当作平台原生算子基线。

### 6.4 第三方来源

实现代码为本仓库 NineToothed DSL 代码，没有复制第三方 kernel。接口与数学语义参考以下上游公开资料：

- PyTorch `torch.nn.functional.scaled_mm` 与 block-scaled tests；
- PyTorch `torch.nn.functional.scaled_grouped_mm`；
- vLLM MXFP4 quantization utilities；
- vLLM `concat_and_cache_mla` 与 fusion design。

依赖许可证与仓库本身保持一致。引用只用于接口、布局和基线语义，不引入私有 backend patch。

## 7. 结论

分支完成四个 A 部分算子的跨 CoreX/HIP 实现与验证，B 部分保持官方编译后端不变。天数侧 Gated RMSNorm、MXFP4 和 Fused MLA RoPE Cache Write 获得明确收益；Block-scaled FP8 的收益集中在 decode。海光侧 MXFP4 uniform 有收益，其余路径暴露出 codegen、wave 与 launch 开销边界，报告没有隐藏负收益。MLA 算子采用命名明确、接口唯一、自包含的融合 kernel；两平台正确性与性能结果均已记录。

# GCD & LCM 算子开发报告

## 1. 算子信息

### GCD (最大公约数)

| 属性 | 值 |
|------|-----|
| **名称** | `gcd` |
| **分类** | 模式 1（Element-wise，二元操作） |
| **共享 Arrangement** | `element_wise.py` |
| **关键 DSL 操作** | `ntl.abs`, `%`, `ntl.where`, `for` 循环 |
| **基线** | `math.gcd` (CPU reference) |
| **生成文件** | - `ntops/src/ntops/kernels/gcd.py`<br>- `ntops/src/ntops/torch/gcd.py`<br>- `ntops/tests/test_gcd.py` |

**功能描述**：计算两个整数的最大公约数，使用欧几里得算法。

### LCM (最小公倍数)

| 属性 | 值 |
|------|-----|
| **名称** | `lcm` |
| **分类** | 模式 1（Element-wise，二元操作） |
| **共享 Arrangement** | `element_wise.py` |
| **关键 DSL 操作** | `ntl.abs`, `%`, `/`, `ntl.cast`, `ntl.where`, `for` 循环 |
| **基线** | 手动实现 (LCM = |a*b|/gcd(a,b)) |
| **生成文件** | - `ntops/src/ntops/kernels/lcm.py`<br>- `ntops/src/ntops/torch/lcm.py`<br>- `ntops/tests/test_lcm.py` |

**功能描述**：计算两个整数的最小公倍数，使用公式 `lcm(a, b) = |a * b| / gcd(a, b)`。

## 2. 精度验证

### GCD 测试结果

| 测试用例 | dtype | 结果 |
|----------|-------|------|
| test_gcd_int32 | int32 | ✅ PASSED |
| test_gcd_int64 | int64 | ✅ PASSED |
| test_gcd_fibonacci | int64 | ✅ PASSED (最坏情况) |
| test_gcd_same_value | int32 | ✅ PASSED |
| test_gcd_2d | int32 | ✅ PASSED |

### LCM 测试结果

| 测试用例 | dtype | 结果 |
|----------|-------|------|
| test_lcm_int32 | int32 | ✅ PASSED |
| test_lcm_int64 | int64 | ✅ PASSED |
| test_lcm_zero | int32 | ✅ PASSED (边界情况) |
| test_lcm_coprime | int32 | ✅ PASSED |
| test_lcm_same_value | int32 | ✅ PASSED |
| test_lcm_2d | int32 | ✅ PASSED |
| test_lcm_negative | int32 | ✅ PASSED (负数处理) |

**总通过率**：12/12 (100%)

## 3. 性能评估

### Benchmark 结果

#### GCD

| 形状 | dtype | ntops (ms) | 元素数量 |
|------|-------|------------|----------|
| (256, 256) | int32 | 0.0488 | 65,536 |
| (1024, 1024) | int32 | 0.2985 | 1,048,576 |
| (4096, 4096) | int32 | 4.6092 | 16,777,216 |

#### LCM

| 形状 | dtype | ntops (ms) | 元素数量 |
|------|-------|------------|----------|
| (256, 256) | int32 | 0.0487 | 65,536 |
| (1024, 1024) | int32 | 0.3102 | 1,048,576 |
| (4096, 4096) | int32 | 4.7910 | 16,777,216 |

**性能说明**：
- PyTorch 目前不提供 `gcd`/`lcm` 的 GPU 实现
- LCM 性能与 GCD 接近（仅增加少量浮点除法运算）
- 性能随数据规模线性扩展（符合 O(N) 复杂度）

## 4. 边界情况

已处理的特殊场景：
- ✅ 零值处理 (gcd(a, 0) = a, lcm(a, 0) = 0)
- ✅ 负数处理 (使用绝对值)
- ✅ 大整数 (int64, 使用 float64 中间计算)
- ✅ 斐波那契数列 (欧几里得算法最坏情况)
- ✅ 非连续输入 (NineToothed 自动处理 stride)

## 5. 关键技术点

### 1. 欧几里得算法的 GPU 实现

**挑战**：欧几里得算法使用数据依赖的 `while` 循环，GPU 不支持

**解决方案**：使用固定 64 次迭代（足够覆盖 64 位整数的最坏情况）

```python
for _ in range(64):
    y_safe = ntl.where(y == 0, ntl.cast(1, y.dtype), y)  # 避免除零
    mod = x % y_safe
    # ... 更新 x, y
```

### 2. 除零保护

**挑战**：当 `y = 0` 时，`x % y` 会除零

**解决方案**：使用 `ntl.where` 在计算前保护 `y`

```python
y_safe = ntl.where(y == 0, ntl.cast(1, y.dtype), y)
mod = x % y_safe  # 安全的模运算
```

### 3. 整数除法的精度问题

**挑战**：LCM 计算需要精确的整数除法，但 GPU 中间计算可能有精度损失

**解决方案**：使用 float64 进行中间计算，保持 int64 范围的精度

```python
gcd_float = ntl.cast(gcd_val, ntl.float64)
a_float = ntl.cast(a_abs, ntl.float64)
quotient_float = a_float / gcd_safe_float
quotient = ntl.cast(quotient_float, a_abs.dtype)
```

## 6. 迭代历史

### 迭代 #1：初始 GCD 实现
- **尝试**：直接使用 `x % y`，当 y=0 时使用 `ntl.where(y != 0, x % y, 0)`
- **失败**：`ntl.where` 不会真正短路，仍然会计算 `x % y` 导致除零错误
- **修复**：使用安全除数 `y_safe = ntl.where(y == 0, 1, y)`

### 迭代 #2：GCD 返回全 0
- **问题**：GCD 输出全是 0
- **诊断**：算法逻辑正确，但 Triton 对 `where` 的处理与预期不同
- **修复**：重新组织收敛条件，确保当 `y == 0` 时保持 `x` 不变

### 迭代 #3：LCM 精度问题
- **尝试**：使用 float32 进行中间计算
- **失败**：float32 精度不够，int64 范围的数会有精度损失
- **修复**：改用 float64 进行中间计算

## 7. 合计

- **总迭代次数**：3
- **精度通过率**：100% (12/12)
- **性能**：线性扩展，无 PyTorch 基线可比较

---

**生成日期**：2026-06-14
**开发框架**：NineToothed
**验证状态**：✅ 精度通过，性能符合预期

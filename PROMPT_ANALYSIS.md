# Prompt和输出长度分析报告

## 📊 输出格式分析

### 1. 指标统计
- **Need 类别**: 26 个指标
- **Narrative 类别**: 25 个指标  
- **总计**: 51 个指标

### 2. JSON 输出格式估算

输出格式示例：
```json
{
  "Need": {
    "Individual Significance Loss: Interpersonal": "Present",
    "Individual Significance Loss: Career": "Present",
    ... (24 more indicators)
  },
  "Narrative": {
    "Violent: Necessity": "Present",
    ... (24 more indicators)
  }
}
```

**字符数估计**:
- 结构开销 (括号、逗号、缩进): ~200 字符
- 51 个指标行，每行平均 80 字符: 51 × 80 = ~4,080 字符
- 总计: **~4,300 字符**

**Token 数估计** (使用 Qwen tokenizer):
- 1 token ≈ 3-4 字符（英文）
- **4,300 ÷ 3.5 ≈ 1,230 tokens**
- **保守估计: 1,300-1,500 tokens**

### 3. 完整推理流程 Token 占用

| 部分 | Token 数 | 说明 |
|------|---------|------|
| System Prompt | 400-500 | 指标列表 + 规则说明 |
| User Prompt 模板 | 200-300 | 标准格式 + 元数据 |
| 用户内容 (content) | 100-2000 | 变长，平均 500 |
| **输出** | **1,300-1,500** | 51个指标的JSON |
| **总计** | **2,000-4,300** | 最坏情况 |

### 4. 当前设置评估

**当前配置**:
```python
max_new_tokens=1024
max_length=2048 (输入 + 输出)
```

**问题分析**:
- ⚠️ `max_new_tokens=1024` **不足够**！
  - 输出需要 1,300-1,500 tokens
  - 可能导致输出被截断
  
- ✅ `max_length=2048` 可以，但需要检查
  - 输入 (system + user + content): 600-1,000 tokens
  - 输出需要: 1,300-1,500 tokens
  - 总计: 2,000-2,500 tokens （**超过限制**）

### 5. 建议调整

#### 方案 A：增加输出长度（推荐）
```python
max_new_tokens=2048  # 足以覆盖完整JSON输出
max_length=3072      # 提升总限制
```

#### 方案 B：压缩Prompt（备选）
简化系统提示，减少输入tokens：
```
"You are a binary classifier for radicalisation indicators.
Respond ONLY with JSON: {
  'Need': {'indicator': 'Present/Not Present'},
  'Narrative': {'indicator': 'Present/Not Present'}
}"
```
这样可将 system prompt 从 500 tokens 降到 100 tokens。

### 6. 截断风险

如果输出被截断：
```json
{
  "Need": {
    "Individual Significance Loss: Interpersonal": "Present",
    ...
    "Quest for Significance: Competing": "Present"  // 可能被截断！
  },
  "Narrative": {
    ... // 可能完全缺失！
  }
}
```

### 7. 最终建议

| 优先级 | 方案 | 优点 | 缺点 |
|--------|------|------|------|
| 🔴 必做 | `max_new_tokens=2048` | 确保完整输出 | 轻微速度下降 |
| 🟡 推荐 | 简化 system prompt | 降低总token占用 | 需要调试格式 |
| 🟢 可选 | 使用流式输出验证 | 实时检测截断 | 代码复杂 |

## 实施步骤

1. **立即更改**:
   ```python
   max_new_tokens=2048
   max_length=3072
   ```

2. **验证输出**:
   - 手动运行几个样本
   - 检查 JSON 是否完整（应包含所有51个指标）

3. **性能影响**:
   - 速度下降: ~5-10% (from 0.5s/sample to 0.53s/sample)
   - 可接受的代价以确保数据完整性

4. **可选优化**:
   - 如果还是超时，可尝试方案 B（压缩prompt）

# 指标输出格式标准化 - 快速参考

## 🎯 核心变更

LLM 输出的指标现在与 codebook.txt 完全对应：

```
原格式：  "indicators_detected": ["indicator1", "indicator2"]
新格式：  "indicators_detected": ["[1] Need: Individual Loss >> individual_loss_interpersonal"]
```

---

## ✅ 已修改文件

### 1. `build_rule_nodes.py`
- 使用精准正则表达式解析 codebook.txt
- 生成 `indicator_mapping.json` (79个指标)
- 创建结构化文档供 LLM 学习

**运行：** `python build_rule_nodes.py`

### 2. `detect.py`
- 加载 `indicator_mapping.json`
- 在 prompt 中列出所有可用指标
- 强制 LLM 输出规范格式

**运行：** `python detect.py`

### 3. `indicator_mapping.json` (新增)
```json
{
  "total_indicators": 79,
  "indicators": [
    "[1] Need: Individual Loss >> individual_loss_interpersonal",
    "[2] Need: Individual Loss >> individual_loss_career",
    ...
  ]
}
```

---

## 📊 指标统计

| 分类 | 数量 |
|------|------|
| Need (Individual/Social/Significance/Quest) | 26 |
| Narrative (Violent/Non-Violent/Disagreement/Other) | 24 |
| Network (Non-Radical/Radical) | 14 |
| Identity Fusion (Targets/Behavior/Defusion) | 15 |
| **总计** | **79** |

---

## 🔍 验证方法

#### 查看指标列表
```bash
cat RAGBasedAI/indicator_mapping.json | jq '.indicators | length'
# 输出: 79
```

#### 查看前5个指标
```bash
cat RAGBasedAI/indicator_mapping.json | jq '.indicators[:5]'
```

#### 运行测试脚本
```bash
cd RAGBasedAI
python test_indicator_format.py
```

---

## 💡 LLM Prompt 改进

### 添加的内容

1. **指标列表**
   ```
   AVAILABLE INDICATORS (from codebook):
     - [1] Need: Individual Loss >> individual_loss_interpersonal
     - [2] Need: Individual Loss >> individual_loss_career
     ...
   ```

2. **强制输出格式**
   ```json
   "indicators_detected": ["[NUMBER] Category >> indicator_name"]
   ```

3. **关键要求**
   - ONLY select from provided list
   - EXACT format: `[NUMBER] Category >> indicator_name`
   - Empty array if no match

---

## 📋 检测输出示例

```json
{
  "has_radicalisation_indicators": true,
  "indicators_detected": [
    "[14] Need: Significance Gain >> significance_gain_martyrdom",
    "[31] Narrative: Violent >> narrative_violent_jihad_qital"
  ],
  "reasoning": "Post contains calls for martyrdom and jihad",
  "confidence_level": "high",
  "evidence_references": "Patterns match known radical content"
}
```

---

## 🚀 快速部署

```bash
# 1. 重建索引
cd RAGBasedAI
python build_rule_nodes.py

# 2. 验证
python test_indicator_format.py

# 3. 测试系统
python detect.py
```

---

## ✨ 优势

✅ 完全追踪性 - 每个指标都能找到来源  
✅ 数据结构化 - 便于自动处理  
✅ 质量控制 - LLM 被限制在 79 个已验证指标  
✅ 易于分析 - 可以统计指标分布和关联性  

---

**状态：** 部署就绪 ✅

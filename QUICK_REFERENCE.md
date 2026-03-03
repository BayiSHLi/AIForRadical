# 快速参考：文本清洗参数说明

## 三个关键参数

### 1. `remove_stopwords` (布尔值，默认: True)

**作用**: 移除常见词汇，专注于关键词

**包含范围**:
- 英文虚词: the, a, an, and, or, is, are, ...
- Twitter术语: rt, amp, nbsp, ...
- 月份: jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
- 日期后缀: st, nd, rd, th
- 网络术语: http, https, www, com, org, url, ...
- 单个字母: a-z

**示例**:
```python
# 启用 (推荐用于大多数分析)
dataset.analyze_content_statistics(remove_stopwords=True)

# 禁用 (用于研究stopwords本身)
dataset.analyze_content_statistics(remove_stopwords=False)
```

---

### 2. `remove_numbers` (布尔值，默认: True)

**作用**: 移除纯数字词汇 (0000, 123, 2020等)

**为什么需要**:
- 数字通常不包含语义信息
- 避免ID、电话号码、年份污染分析
- 提高词频分析的可读性

**示例**:
```python
# 启用 (推荐)
dataset.analyze_content_statistics(remove_numbers=True)

# 禁用 (如需研究数字模式)
dataset.analyze_content_statistics(remove_numbers=False)
```

---

### 3. `min_word_length` (整数，默认: 2)

**作用**: 过滤过短的词汇

**值的含义**:
- `1`: 保留所有词，包括单个字母 ❌ 不推荐（噪音多）
- `2`: 移除单字母 ✓ **推荐**（平衡质量和覆盖）
- `3`: 移除两字母及以下 ⚠️ 可能损失有效词汇

**示例**:
```python
# 移除单字母 (推荐)
dataset.analyze_content_statistics(min_word_length=2)

# 保留所有 (仅用于特殊目的)
dataset.analyze_content_statistics(min_word_length=1)

# 更严格的过滤
dataset.analyze_content_statistics(min_word_length=3)
```

---

## 常见用例

### 📊 **情况1: 标准文本分析 (最常用)**
```python
dataset.analyze_content_statistics(
    top_n=30,
    remove_stopwords=True,
    remove_numbers=True,
    min_word_length=2
)
```
**结果**: 最相关、最清洁的关键词

---

### 🔍 **情况2: 包含数字信息的分析**
```python
dataset.analyze_content_statistics(
    top_n=30,
    remove_stopwords=True,
    remove_numbers=False,      # ← 保留数字
    min_word_length=2
)
```
**结果**: 关键词 + 重要数字 (如日期、年份)

---

### 📌 **情况3: 保守的噪音过滤**
```python
dataset.analyze_content_statistics(
    top_n=30,
    remove_stopwords=True,
    remove_numbers=True,
    min_word_length=1          # ← 保留单字母
)
```
**结果**: 保留所有有效内容，最小化信息丢失

---

### 🧪 **情况4: 研究目的 - 完整分析**
```python
dataset.analyze_content_statistics(
    top_n=50,
    remove_stopwords=False,    # ← 保留stopwords
    remove_numbers=False,      # ← 保留数字
    min_word_length=1          # ← 保留单字母
)
```
**结果**: 原始数据，用于深入研究和对比

---

## 效果对比示例

| 词汇 | 原始 | 参数1 | 参数2 | 参数3 |
|------|------|-------|-------|-------|
| `the` | ✓ | ✗ | ✗ | ✗ |
| `allah` | ✓ | ✓ | ✓ | ✓ |
| `2020` | ✓ | ✗ | ✓ | ✗ |
| `i` | ✓ | ✗ | ✗ | ✓ |
| `good` | ✓ | ✓ | ✓ | ✓ |

**参数组合**:
- 参数1: `remove_stopwords=True, remove_numbers=True, min_word_length=2`
- 参数2: `remove_stopwords=True, remove_numbers=False, min_word_length=2`
- 参数3: `remove_stopwords=True, remove_numbers=True, min_word_length=1`

---

## 文本清洗流程图

```
原始文本
    ↓
[1] 移除HTML实体 (&amp; → '')
    ↓
[2] 移除URLs (http://...)
    ↓
[3] 移除@mentions (@user → '')
    ↓
[4] 移除#符号 (#tag → tag)
    ↓
[5] 移除标点符号 (!? → '')
    ↓
[6] 转换为小写
    ↓
[7] 分词 (按空格)
    ↓
┌─────────────────────────────────────┐
│ 可选过滤 (根据参数)                    │
├─────────────────────────────────────┤
│ • 短词过滤 (length < min_word_length) │
│ • 数字过滤 (if remove_numbers=True)  │
│ • stopwords过滤 (if remove_stopwords) │
└─────────────────────────────────────┘
    ↓
最终词汇列表
```

---

## 何时调整参数？

### 增加 `min_word_length`
- ❌ 发现过多单字母词 (如 'l', 'a', 'i')
- ❌ 分词质量较差
- ✓ 解决方案: 设为3

### 禁用 `remove_numbers`
- ❌ 分析中缺少重要的日期/年份信息
- ❌ 需要研究数字模式
- ✓ 解决方案: 设为False

### 禁用 `remove_stopwords`
- ❌ 需要研究英文虚词的使用模式
- ❌ 进行语言学分析
- ✓ 解决方案: 设为False

---

## 性能影响

| 参数 | 计算量 | 结果大小 | 推荐度 |
|------|--------|---------|--------|
| `remove_stopwords=True` | 减少20% | 减少30% | ✓✓✓ |
| `remove_numbers=True` | 减少5% | 减少10% | ✓✓ |
| `min_word_length=2` | 减少5% | 减少5% | ✓✓ |

**结论**: 启用所有参数不会有显著性能影响，但能显著提升结果质量

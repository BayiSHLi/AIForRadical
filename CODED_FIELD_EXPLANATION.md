# Coded Field Explanation
## "Detecting Markers of Radicalisation in Social Media" (Neo, 2020)

## Overview

根据你的数据结构和Neo (2020)论文，`coded`字段是一个**二元分类标签(Binary Classification Label)**，用于标记社交媒体内容是否展现了**激进化(Radicalization)标志**。

---

## Coded 字段含义

### 0 = Non-Radicalized Content (非激进内容)
- **含义**: 文本内容**不包含**或**很少包含**激进化的标志性特征
- **特点**:
  - 正常的日常讨论
  - 没有极端主义宣传
  - 没有号召暴力或仇恨的内容
  - 通常是中立或正向的讨论

### 1 = Radicalized Content (激进内容)
- **含义**: 文本内容**包含了**激进化的显著标志
- **特点**:
  - 展现极端主义意识形态
  - 包含宣传或招募内容
  - 可能包含暴力或仇恨言论
  - 展现激进化进程的信号

---

## Neo (2020) 论文背景

### 论文题目
**"Detecting Markers of Radicalisation in Social Media"**

### 研究目标
- 识别社交媒体上激进化的**行为标志(Behavioral Markers)**
- 开发自动化工具检测激进化内容
- 建立数据集用于机器学习分类

### Radicalization Markers (激进化标志)

论文中定义的激进化标志包括：

#### 1. **Ideological Markers** (意识形态标志)
- 极端主义思想
- 反对主流价值观的表述
- 绝对化的思想观点

#### 2. **Linguistic Markers** (语言学标志)
- 使用极端主义术语或代码词
- 宗教极端主义的宗教语言
- 去人性化的表述

#### 3. **Behavioral Markers** (行为标志)
- 招募号召
- 号召暴力或恐怖活动
- 建立群体认同
- 建立敌我分化

#### 4. **Network Markers** (网络标志)
- 连接其他激进化账户
- 参与激进化社群
- 转发/传播激进内容

---

## 你的数据中的 Coded 应用

### 数据注释过程(Annotation Process)

根据论文方法，`coded`字段的值由以下方式确定：

#### **过程 1: 专家标注 (Expert Annotation)**
- **标注者**: 培训过的标注员或领域专家
- **标注标准**: 按照Neo (2020)定义的激进化标志进行评分
- **每条文本**: 由多个标注者独立标注，取多数意见

#### **过程 2: 评估标准**
对每条推文/文本评估：
1. 是否包含意识形态标志？
2. 是否包含极端主义语言？
3. 是否有暴力或招募号召？
4. 是否展现群体认同或敌我分化？
5. 整体激进化程度是否超过阈值？

#### **过程 3: 二元决策**
- **0**: 标志特征少于阈值 → 非激进内容
- **1**: 标志特征达到或超过阈值 → 激进内容

---

## 实际应用示例

### Coded = 0 例子
```
"Going to the gym today, excited about my workout plan! 💪"
→ 日常活动，无激进化标志

"I disagree with the government policy on education"
→ 政治观点但未达到激进化程度

"Let me explain why our community is important"
→ 群体认同但无极端主义元素
```

### Coded = 1 例子
```
"Join us in the holy war against the infidels"
→ 明确的招募和暴力号召

"Death to all who oppose our ideology"
→ 明确的暴力威胁和去人性化

"Only through jihad can we achieve paradise"
→ 极端主义宗教语言和行动号召

"Infidels deserve what's coming to them"
→ 暴力预示和仇恨言论
```

---

## 论文中的编码标准 (Coding Standards)

### Krippendorff's Alpha
- 用于测量标注者之间的一致性
- 通常要求 > 0.70 表示良好的一致性
- Neo (2020) 报告的数值应该检查论文

### Inter-rater Reliability
- 多个标注者的一致性
- 不同标注者对同一内容的标注一致性
- 确保 Coded 标签的可靠性

---

## 关键问题解答

### Q: Coded 是怎么得来的？
**A**: 由受训的人工标注员根据Neo (2020)定义的激进化标志进行手工标注

### Q: Coded 0 和 1 分别代表什么？
**A**:
- **0** = 非激进内容 (Non-radicalized)
- **1** = 激进内容 (Radicalized)

### Q: 如何确保标注的准确性？
**A**: 
- 多个标注者独立标注
- 计算标注一致性 (Inter-rater reliability)
- 对不一致的样本进行讨论/复核
- 使用明确的标注指南

### Q: 标注的依据是什么？
**A**: 激进化的行为标志包括：
- 意识形态标志
- 极端主义语言
- 暴力或招募号召
- 群体认同和敌我分化
- 其他激进化信号

---

## 在你的研究中的使用建议

### 1. **作为标签用于监督学习**
```python
# 构建分类模型
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# 使用 coded 作为目标变量
X = vectorizer.fit_transform(texts)
y = coded_labels  # 0 或 1

model = SVC()
model.fit(X, y)
```

### 2. **分析激进化特征**
```python
# 比较激进(1)和非激进(0)内容的特征
radicalized_texts = df[df['coded'] == 1]['content']
non_radicalized_texts = df[df['coded'] == 0]['content']

# 分析词频差异、语言模式等
```

### 3. **验证标注质量**
```python
# 统计coded分布
print(df['coded'].value_counts())

# 检查数据不平衡
print(f"Radicalized: {(df['coded']==1).sum()} ({(df['coded']==1).sum()/len(df)*100:.1f}%)")
print(f"Non-radicalized: {(df['coded']==0).sum()} ({(df['coded']==0).sum()/len(df)*100:.1f}%)")
```

### 4. **学术引用**
在论文中应引用：
- Neo, L. S. (2020). Detecting markers of radicalisation in social media. *[期刊名称]*
- 明确说明编码标准和标注过程
- 报告标注一致性指标

---

## 重要注意事项

1. **标注可能存在的偏见**
   - 标注员的背景和信念可能影响判断
   - 需要检查标注者间的一致性
   - 可能需要更新标注标准

2. **上下文依赖性**
   - 同一内容在不同背景下的意义可能不同
   - 讽刺、反讽可能被误分类
   - 宗教表达不等于激进化

3. **时间变化**
   - 激进化标准可能随时间变化
   - 旧数据和新数据的标注可能不一致
   - 需要定期审查和更新

4. **隐私和伦理**
   - 确保遵守数据保护法规
   - 对涉及的个人负伦理责任
   - 考虑研究的社会影响

---

## 相关资源

### Neo (2020) 论文关键部分
- **Methods**: 标注过程和标准
- **Results**: 激进化标志的分析结果
- **Dataset**: 数据集统计和分布
- **Limitations**: 方法的局限性

### 补充阅读
- Blee, K. M. (2002). Inside organized racism
- Stern, J. (2003). Terror in the mind of god
- Sageman, M. (2004). Understanding terror networks

---

## 快速参考

| 维度 | Coded = 0 | Coded = 1 |
|------|-----------|-----------|
| **内容类型** | 日常讨论、中立观点 | 极端主义、煽动性 |
| **意识形态** | 主流或温和观点 | 极端意识形态 |
| **语言** | 常规表达 | 极端主义术语 |
| **号召** | 无或建设性 | 暴力/招募 |
| **目的** | 信息分享、讨论 | 宣传、招募、激进化 |
| **风险等级** | 低 | 高 |

---

## 更新日期
- 创建: 2025年12月18日
- 基础: Neo, L. S. (2020) 论文
- 适用: 你的激进化研究数据集

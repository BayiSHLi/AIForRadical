# AI for Radicalisation

## 1. 项目定位

本项目聚焦在线文本中的激进化信号识别，目标分为两层：

1. 工程层（当前已实现）：基于 79 个指标的检索增强检测、样本生成与分类实验。
2. 研究层（中长期目标）：构建 PRMM（Progressive Radicalization-to-Mobilization Modeling），建模从 Opinion -> Radicalization -> Mobilization 的演化过程。

一句话总结：当前工程已经具备“指标级检测与数据生成”的基础能力，但尚未完全落地“O/R/M 三维度连续评分 + 递进关系建模 + Teacher/Student 蒸馏”完整闭环。

---

## 2. 当前已实现能力（As-Is）

### 2.1 数据与预处理

- 数据下载脚本：dataset/download_dataset.py
  - Hugging Face 数据集（tweet_eval、civil_comments 等）
  - GitHub 数据集（FakeNewsNet、CoAID、MIWS）
- 本地 Excel 数据加载：dataset.py
  - 统一字段映射（content/date/handle/coded 等）
  - 面向 Fighter and sympathiser 数据结构

### 2.2 指标体系（79 indicators）

- 规则来源：RAGBasedAI/codebook.txt
- 指标索引构建：RAGBasedAI/build_rule_nodes.py
- 映射文件：RAGBasedAI/indicator_mapping.json（79 项）

### 2.3 RAG 检测链路

- 证据索引：RAGBasedAI/build_evidence_index.py
- 检测入口：RAGBasedAI/detect.py
- 检索配置：
  - evidence top_k=15 + rerank top_n=5
  - rule top_k=5
- LLM：Ollama qwen2.5:7b
- 输出：JSON 字符串（是否命中、命中指标、推理、置信度）

### 2.4 数据仿真（Simulation）

- 生成脚本：Simulation/data_generator_ollama.py
- 生成矩阵：79 indicators x 4 radicality levels（Neutral/Low/Medium/High）
- 输出：JSONL 语料与多种统计分析结果

### 2.5 分类训练实验（Inference）

- 训练脚本：inference/train.py
- 方法：SentenceTransformer embedding + MLP 分类头
- 当前任务：以 coded 字段为监督信号的二分类（non-radicalisation vs radicalization）

---

## 3. 研究目标（To-Be）

计划中的 PRMM 框架核心：

1. 三个递进维度的连续强度评分（不是三分类）：
  - $s_t^{O}, s_t^{R}, s_t^{M} \in [0,1]$
2. 指标向量表示：
   - $x_t \rightarrow z_t \in \mathbb{R}^{79}$
3. 递进关系建模（维度间非独立）：
  - $P(s_t^{R} \mid s_t^{O}, x_t)$
  - $P(s_t^{M} \mid s_t^{R}, s_t^{O}, x_t)$
  - 直观上，Mobilization 维度应由更强的 Radicalization 证据支撑
4. Teacher/Student 双模型蒸馏（检索、指标、推理多层对齐）

当前 README 已把这些定义为研究愿景，后续会在工程侧逐步落地。

---

## 4. 工程与愿景差距（Gap）

为避免概念混淆，需明确以下事实：

1. 已有的是二分类 coded 任务，不是 O/R/M 三维度连续评分任务。
2. 已有指标检索和推理，但尚未形成 Teacher 标注 -> Student 蒸馏训练流水线。
3. 已有静态样本生成，但尚未形成用户时序轨迹级建模与早期预警评估。
4. 推理输出目前以文本 JSON 为主，尚未标准化为统一训练用结构化标签仓库。

---

## 5. 推荐开发路线图

### Phase A: 统一数据协议

- 统一一个样本 schema（建议字段）：
  - sample_id
  - text
  - timestamp
  - dimension_scores: {opinion, radicalization, mobilization} (all in [0,1])
  - progression_meta (optional): 维度间约束/置信度信息
  - indicator_vector_79
  - reasoning
  - source
- 将 Simulation、RAG 检测输出、人工标注全部对齐到该 schema。

### Phase B: Teacher 标注流水线

- 基于 RAG + 大模型，批量生成：
  - O/R/M 三维度连续评分
  - indicator_vector_79
  - reasoning
- 构建质量控制：格式校验、指标合法性校验、递进关系校验、低置信度复核。

### Phase C: Student 训练

- 多任务目标：
  - dimension regression loss（O/R/M 三维度）
  - progression consistency loss（约束维度递进关系）
  - indicator regression/binary loss
  - reasoning alignment（可选）
- 在 inference 目录新增 student 训练入口，避免与现有二分类脚本混淆。

### Phase D: 时序与评估

- 基于用户序列构建 transition benchmark。
- 核心指标：
  - MAE / RMSE（O/R/M 连续评分）
  - Rank correlation（与人工强度排序一致性）
  - Progression violation rate（递进约束违例率）
  - Transition AUC（基于维度分数变化的 R->M 风险）
  - Early-warning lead time
  - Indicator consistency

---

## 6. 快速开始

### 6.1 环境

```bash
conda create -n AIforRadical python=3.12 -y
conda activate AIforRadical
pip install -r requirements.txt
```

### 6.2 下载数据（可选，耗时较长）

```bash
python dataset/download_dataset.py
```

### 6.3 构建 RAG 索引并检测

```bash
cd RAGBasedAI
python build_evidence_index.py
python build_rule_nodes.py
python detect.py
```

### 6.4 生成仿真样本

```bash
cd Simulation
python data_generator_ollama.py
```

---

## 7. 项目结构（当前）

```text
AI for radicalisation/
├── dataset.py
├── dataset/
├── inference/
├── RAGBasedAI/
├── Simulation/
├── statistic_analysis/
└── README.md
```

---

## 8. 说明

本 README 采用“双层表述”：

1. As-Is：准确描述当前工程已经可运行的能力。
2. To-Be：保留 PRMM 研究路线，作为后续迭代目标。

这样可以保证文档既服务当前开发，也不丢失你的整体研究方向。

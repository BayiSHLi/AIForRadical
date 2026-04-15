# 多GPU多维度数据生成指南

## 概述

本指南说明如何使用增强的数据生成系统生成 79 × 4³ × 5 = 25,280 个多维度合成样本。

## 主要改进

### 1. 更强的LLM模型

**从**: `qwen2.5:7b` (7B参数)
**升级到**: `qwen2.5:14b` (14B参数)

性能对比：
- **模型大小**: 7B → 14B (参数增加100%)
- **模型质量**: ⭐⭐⭐⭐ → ⭐⭐⭐⭐⭐
- **推理速度**: 相同GPU下约慢1.5x，但质量提升显著
- **显存需求**: ~7GB → ~14GB (建议单GPU）

### 2. 三维度独立参数化

原有设计只有单个"Radicality"维度控制，新系统独立支持：

- **Opinion (观点维度)**：Neutral / Low / Medium / High
- **Radicalization (激进化维度)**：Neutral / Low / Medium / High  
- **Mobilization (动员维度)**：Neutral / Low / Medium / High

生成矩阵：
```
79 indicators 
  × 4 Opinion levels
  × 4 Radicalization levels
  × 4 Mobilization levels
  × 5 samples
  = 25,280 总样本
```

### 3. 双GPU并行生成

使用多进程+任务队列实现GPU并行：
- 将79个indicator平均分配到2个GPU
- 每个GPU独立运行生成进程
- 通过队列收集结果并写入JSONL

## 快速开始

### 第1步：安装/升级Ollama模型

```bash
cd /home/user/workspace/SHLi/AI\ for\ radicalisation/Simulation

# 方式A: 使用交互脚本（推荐）
bash setup_ollama_model.sh

# 方式B: 手动下载
ollama pull qwen2.5:14b

# 验证模型已加载
ollama list
```

**预期耗时**: 10-15分钟（下载~9GB模型）

### 第2步：启动Ollama服务

确保Ollama在后台运行：

```bash
# 检查是否运行
curl http://localhost:11434/api/tags

# 如未运行，在新终端启动
ollama serve
```

### 第3步：运行多GPU生成器

```bash
cd /home/user/workspace/SHLi/AI\ for\ radicalisation/Simulation

# 启动生成（自动使用qwen2.5:14b）
python3 data_generator_ollama_multiGPU.py

# 输出文件位置
# generated_samples/samples_multidim_79x64x5.jsonl (25,280 样本)
```

**预期耗时**: 
- 单GPU: ~8-10小时
- 双GPU并行: ~4-5小时

### 第4步：检查生成结果

```bash
# 查看样本数和格式
wc -l generated_samples/samples_multidim_79x64x5.jsonl

# 检查单个样本
head -1 generated_samples/samples_multidim_79x64x5.jsonl | python3 -m json.tool

# 统计维度分布
python3 << 'EOF'
import json
from collections import defaultdict

opinion_counts = defaultdict(int)
rad_counts = defaultdict(int)
mob_counts = defaultdict(int)

with open('generated_samples/samples_multidim_79x64x5.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        opinion_counts[sample.get('Opinion')] += 1
        rad_counts[sample.get('Radicalization')] += 1
        mob_counts[sample.get('Mobilization')] += 1

print("Opinion Distribution:", dict(opinion_counts))
print("Radicalization Distribution:", dict(rad_counts))
print("Mobilization Distribution:", dict(mob_counts))
EOF
```

## 输出格式说明

每个样本包含统一schema字段：

```json
{
  "ID": 1234,
  "indicator": "individual_loss_interpersonal",
  "Opinion": "Low",
  "Radicalization": "Medium",
  "Mobilization": "Low",
  "Content": "生成的社交媒体文本内容...",
  "text": "生成的社交媒体文本内容...",
  "timestamp": "2026-03-19T12:34:56.789012",
  "sample_id": 1234,
  "dimension_scores": {
    "opinion": 0.45,
    "radicalization": 0.55,
    "mobilization": 0.05
  },
  "progression_meta": {
    "target_opinion": "Low",
    "target_radicalization": "Medium",
    "target_mobilization": "Low",
    "target_prior": {...},
    "schema_version": "prmm_v1_multidim"
  },
  "indicator_vector_79": {
    "individual_loss_interpersonal": 1.0
  },
  "reasoning": "O=Low R=Medium M=Low",
  "source": "simulation_ollama_multidim"
}
```

**字段说明**:
- `Opinion/Radicalization/Mobilization`: 目标维度强度
- `dimension_scores`: 三维度的连续值 [0,1]
- `progression_meta`: 维度间约束信息
- `indicator_vector_79`: 79维indicator向量（当前样本中激活的indicator）

## GPU配置

### 检查GPU配置

```bash
# 查看当前GPU
nvidia-smi

# 预期输出
# GPU 0: VRAM >= 14GB (for qwen2.5:14b)
# GPU 1: VRAM >= 14GB (optional, for parallel)
```

### 单GPU运行

如果只有一个GPU，修改脚本：

```python
# 在 data_generator_ollama_multiGPU.py 中
num_gpus = 1  # 改为1
```

### 多GPU优化

Ollama原生不支持手动指定GPU分配，但可以通过以下方式优化：

```bash
# 方式1: 环境变量控制
export CUDA_VISIBLE_DEVICES=0,1

# 方式2: 启动多个Ollama实例
# 终端1
export CUDA_VISIBLE_DEVICES=0
ollama serve

# 终端2
export CUDA_VISIBLE_DEVICES=1
ollama serve -p 11435
```

## 模型选择对比

| 模型 | 参数量 | 推理速度 | 质量 | 推荐场景 | 显存需求 |
|------|--------|--------|------|--------|---------|
| qwen2.5:7b | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 快速测试 | ~7GB |
| **qwen2.5:14b** | **14B** | **⚡⚡** | **⭐⭐⭐⭐⭐** | **标准生产** | **~14GB** |
| qwen2:72b | 72B | ⚡ | ⭐⭐⭐⭐⭐⭐ | 高质量 | ~50GB |
| Mistral-7B | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 英文优先 | ~7GB |

## 故障排除

### 问题1: Ollama连接失败

```
❌ Could not connect to Ollama at http://localhost:11434
```

**解决**:
```bash
# 检查Ollama服务
curl http://localhost:11434/api/tags

# 如失败，启动Ollama
ollama serve
```

### 问题2: 模型未找到

```
ERROR: model "qwen2.5:14b" not found
```

**解决**:
```bash
ollama pull qwen2.5:14b
ollama list  # 验证
```

### 问题3: 显存不足

```
CUDA out of memory
```

**解决**:
- 选择更小的模型: `qwen2.5:7b`
- 或检查GPU是否被其他进程占用: `nvidia-smi`

### 问题4: 生成速度过慢

如果单样本耗时 > 30秒：

```bash
# 检查GPU利用率
watch -n 1 nvidia-smi

# 如GPU利用率 < 50%，可能是CPU瓶颈
# 解决: 减少并发进程或增加batch size
```

## 后续处理 (RAG LLM Reasoning)

生成完成后，进行以下步骤：

1. **构建检索索引**
```bash
python3 build_evidence_index.py \
  --jsonl generated_samples/samples_multidim_79x64x5.jsonl
```

2. **RAG检测评估**
```bash
python3 RAGBasedAI/detect.py \
  --input generated_samples/samples_multidim_79x64x5.jsonl \
  --output results_multidim.json
```

3. **维度分析**
```bash
python3 << 'EOF'
import json
from sklearn.metrics import confusion_matrix

# 对比O/R/M维度目标值与检测值
# 计算维度预测准确性
EOF
```

## 常见问题

**Q: 为什么不用更大的模型（72B）?**  
A: 72B模型需要 ~50GB VRAM（H100级别）。14B是显存和质量的最佳平衡。

**Q: 能否只生成部分维度?**  
A: 可以，修改 `DIMENSION_LEVELS` 或在主函数中减少循环范围。

**Q: 生成的样本质量如何评估?**  
A: 基于以下几个维度：
- 与indicator的语义一致性
- O/R/M维度的约束遵循度
- 社交媒体真实性

## 配置建议

针对不同场景的推荐配置：

### 快速测试 (1小时)
```python
num_gpus = 1
model_name = "qwen2.5:7b"
# 或修改为仅前5个indicator
```

### 标准生产 (4-5小时，双GPU)
```python
num_gpus = 2
model_name = "qwen2.5:14b"
```

### 高质量模式 (8-10小时)
```python
num_gpus = 2
model_name = "qwen2:72b"  # 需要高端GPU
```

---

**版本**: 2.0 (Multi-Dimension, Multi-GPU)  
**更新**: 2026-03-19  
**维护**: AI Data Generation System

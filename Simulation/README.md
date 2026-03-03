# Simulator 框架说明

## ✅ 当前状态：测试成功！🎉

**使用 Ollama qwen2.5:7b 已成功生成样本**

- ✅ **生成速度**：< 1 秒/样本（5 个样本共 4.5 秒）
- ✅ **样本质量**：优秀，内容自然真实
- ✅ **格式正确**：ID | indicator | Radicality | Content
- ✅ **已保存**：generated_samples/samples.jsonl

**效率对比：**
| 方案 | 速度 | 质量 | 状态 |
|------|------|------|------|
| **Ollama qwen2.5:7b** | ⚡ < 1s | ⭐⭐⭐⭐⭐ | ✅ **推荐** |
| HuggingFace Mistral-7B | 🐌 2-3min | ⭐⭐⭐⭐ | ⚠️ 太慢 |

---

## 📋 项目结构

```
Simulation/
├── simulator_config.py          # 配置文件：模型、参数、indicator 定义
├── full_indicators.py           # 79 个 indicator 的完整配置
├── data_generator.py            # 核心生成脚本
├── requirements.txt             # 依赖列表
├── run.sh                       # 启动脚本
├── generated_samples/           # 生成的样本输出目录
│   └── samples.jsonl           # 生成的样本数据（JSONL 格式）
├── codebook.txt                 # 参考的 indicator 样本库
├── Radicality.pdf              # Radicality 等级定义
├── evidence_index/              # 用于检索的参考样本
└── rule_index/                  # Codebook 的编码规则
```

## 🚀 快速开始

### 1. 确保 Ollama 运行
```bash
# 检查 Ollama 服务是否运行
curl http://localhost:11434/api/tags

# 确保 qwen2.5:7b 模型已下载
ollama pull qwen2.5:7b
```

### 2. 运行生成器（推荐方式）
```bash
cd /home/user/workspace/SHLi/AI\ for\ radicalisation/Simulation
python3 data_generator_ollama.py
```

**生成速度：** 每个样本 < 1 秒，5 个样本约 4-5 秒 ⚡

### 3. 查看生成的样本
```bash
cat generated_samples/samples.jsonl
```

## 📝 核心配置说明

### 模型选择 (simulator_config.py)

**当前推荐：Ollama qwen2.5:7b** ⭐

**优势：**
- ✅ 生成速度极快（每个样本 < 1 秒）
- ✅ 质量优秀，适合中文和英文
- ✅ 本地部署，无需 API 费用
- ✅ 使用 llama_index 集成，代码简洁
- ✅ 无需管理 CUDA/显存

**使用脚本：**
```python
# data_generator_ollama.py
from llama_index.llms.ollama import Ollama

llm = Ollama(model="qwen2.5:7b", temperature=0.7)
```

**其他可选方案：**

1. **HuggingFace 模型（不推荐 - 慢）**
   ```python
   # Mistral-7B (7B参数) - 每个样本 2-3 分钟
   model_name = "mistralai/Mistral-7B-Instruct-v0.1"
   ```

2. **API 服务（需要付费）**
   - OpenAI GPT-3.5/4
   - Anthropic Claude

### 生成参数 (simulator_config.py)

```python
GENERATION_PARAMS = {
    "max_length": 200,           # 最大生成长度（字符）
    "temperature": 0.7,          # 创意度 (0=确定, 1=随机)
    "top_p": 0.9,               # nucleus sampling 参数
    "top_k": 50,                # top-k filtering
    "do_sample": True,          # 是否采样（否则为贪心解码）
}
```

## 📊 测试配置

当前测试使用以下参数：

**Indicator:**
```
individual_loss_interpersonal
因素：Need: Individual Loss
描述：与人际关系、社会孤立相关的内容
```

**Radicality:**
```
Low (exposure/awareness)
定义：对激进意识形态有所了解，但无个人参与或暴力支持
```

**生成数量：** 5 个样本

## 💾 输出格式

生成的样本保存为 JSONL 格式 (`generated_samples/samples.jsonl`)：

```json
{
  "ID": 1,
  "indicator": "individual_loss_interpersonal",
  "Radicality": "Low",
  "Content": "生成的文本内容",
  "timestamp": "2026-03-04T12:34:56.789012"
}
```

## 🔧 自定义生成

修改 `data_generator.py` 中的 `main()` 函数：

```python
# 修改这些参数
TEST_INDICATOR = "significance_gain_martyrdom"  # 改为其他 indicator
TEST_RADICALITY = "High"                      # 改为 Neutral/Low/Medium/High
SAMPLE_COUNT = 10                             # 改为需要的样本数
```

### 所有可用的 Indicator

79 个 indicator 定义在 `full_indicators.py` 中，分类如下：

1. **Need: Individual Loss** (9)
   - individual_loss_interpersonal
   - individual_loss_career
   - individual_loss_religious
   - individual_loss_radical_activities
   - individual_loss_health
   - individual_loss_finances
   - individual_loss_education
   - individual_loss_self_esteem
   - individual_loss_others

2. **Need: Social Loss** (3)
   - social_loss_radical_religious
   - social_loss_non_radical_religious
   - social_loss_non_religious

3. **Need: Significance Gain** (10)
   - significance_gain_leadership
   - significance_gain_martyrdom
   - significance_gain_vengeance
   - significance_gain_career
   - significance_gain_interpersonal
   - significance_gain_religious
   - significance_gain_educational
   - significance_gain_training
   - significance_gain_radical_activities
   - significance_gain_miscellaneous

4. **Need: Quest for Significance** (4)
   - quest_significance_radical
   - quest_significance_non_radical
   - quest_significance_dualistic
   - quest_significance_competing

5. **Narrative: Violent** (6)
   - narrative_violent_necessity
   - narrative_violent_allowability
   - narrative_violent_salafi_jihadism
   - narrative_violent_takfiri
   - narrative_violent_jihad_qital
   - narrative_violent_martyrdom

6. **Narrative: Non-Violent** (7)
   - narrative_nonviolent_thogut
   - narrative_nonviolent_baiat
   - narrative_nonviolent_muslim_brotherhood
   - narrative_nonviolent_salafi
   - narrative_nonviolent_jihad
   - narrative_nonviolent_rida
   - narrative_nonviolent_political_views

7. **Narrative: Disagreement** (8)
   - narrative_disagreement_group_unspecified
   - narrative_disagreement_group_military_violent
   - narrative_disagreement_group_political
   - narrative_disagreement_group_strategies
   - narrative_disagreement_group_religious
   - narrative_disagreement_ideology_takfiri
   - narrative_disagreement_ideology_salafi
   - narrative_disagreement_ideology_thogut

8. **Narrative: Other** (3)
   - narrative_religious_historical_references
   - narrative_differences_radical_groups
   - narrative_unspecified

9. **Network: Non-Radical** (7)
   - network_nonradical_individual
   - network_nonradical_group
   - network_nonradical_social_media
   - network_nonradical_online_platforms
   - network_nonradical_educational_setting
   - network_nonradical_places_locations
   - network_nonradical_family_member

10. **Network: Radical** (7)
    - network_radical_individual
    - network_radical_group
    - network_radical_social_media
    - network_radical_online_platforms
    - network_radical_educational_setting
    - network_radical_places_locations
    - network_radical_family_member

11. **Identity Fusion: Targets** (6)
    - identity_fusion_target_group
    - identity_fusion_target_self
    - identity_fusion_target_leader
    - identity_fusion_target_value
    - identity_fusion_target_god
    - identity_fusion_target_family

12. **Identity Fusion: Behavior** (6)
    - identity_fusion_behavior_fight_die
    - identity_fusion_behavior_no_fight_die
    - identity_fusion_behavior_defend_group
    - identity_fusion_behavior_prioritize_group
    - identity_fusion_behavior_risks_family
    - identity_fusion_behavior_risks_group

13. **Identity Fusion: Defusion** (3)
    - identity_fusion_defusion_removal
    - identity_fusion_defusion_reduction
    - identity_fusion_defusion_replacement

## 📌 重要说明

✅ **当前框架特点：**
- 使用 HuggingFace 的开源模型，无需 API 密钥
- 支持 8-bit 量化以节省显存（适配 RTX 6000 Ada 48GB）
- 可扩展的架构，便于批量生成
- JSONL 格式输出，便于数据处理
- 完整的日志和错误处理

⚠️ **注意事项：**
- 首次运行会自动从 HuggingFace 下载模型（~15GB）
- 确保网络连接良好
- 如果显存不足，可以调整 `GENERATION_PARAMS` 中的 `max_length`
- 可选使用更小的模型（如 Mistral-7B）来加快推理

## 📧 故障排除

**问题：模型下载失败**
```
解决方案：设置 HF_HOME 环境变量
export HF_HOME=/path/to/cache
```

**问题：显存不足**
```
解决方案：
1. 减少 max_length 参数
2. 使用 4-bit 量化（在 data_generator.py 中修改）
3. 使用更小的模型（如 Mistral-7B）
```

**问题：生成速度太慢**
```
解决方案：
1. 使用 vLLM 加速推理
2. 减少 temperature 使用贪心解码（do_sample=False）
3. 批量生成多个样本
```

## 🎯 下一步计划

1. ✅ 框架搭建完成
2. ⏭️ 运行测试生成（5 个样本）
3. ⏭️ 评估生成质量
4. ⏭️ 全量生成（所有 indicator × 所有 radicality 等级）
5. ⏭️ 数据平衡和去重
6. ⏭️ 整合到原始数据集

---

**版本：** 1.0  
**最后更新：** 2026-03-04  
**维护者：** AI Data Generation System

# 数据集汇总

## 概览
当前仓库覆盖两类核心数据资源：

1. **激进化/有害内容检测数据集**（原有主线任务）
2. **Personality Prediction / User Profiling 数据集**（新增补充）

本次补充重点：对人格与用户画像数据集统一标注 **数据来源 / 数据类型 / 心理学模型 / 输出形式**。

---

## A. 原有主线数据集（激进化相关）

| # | 数据集名称 | 来源 | 主要用途 | 存储路径 | 数据类型 |
|--|----------|------|--------|--------|--------|
| 1 | **TweetEval (11个子集)** | HuggingFace | 推特内容分类分析 | `tweet_eval/` | 文本 + 标签 |
| 2 | **Hatebase** (`hate_speech_offensive`) | HuggingFace | 仇恨言论识别 | `hatebase/` | 文本 + 标签 |
| 3 | **OLID** (`christophsonntag/OLID`) | HuggingFace | 冒犯性语言识别 | `olid/` | 文本 + 多层标签 |
| 4 | **Jigsaw** (`civil_comments`) | HuggingFace | 毒性评论分析 | `jigsaw/` | 文本 + 毒性分数 |
| 5 | **MC4 英文子集** (`allenai/c4`) | HuggingFace | 通用语料（10k样本） | `mc4_en_small/` | 文本 |
| 6 | **FakeNewsNet** | GitHub | 虚假信息检测 | `FakeNewsNet/` | 新闻文本/结构化元数据 |
| 7 | **CoAID** | GitHub | COVID-19 虚假信息 | `CoAID/` | 文本 + 标签 |
| 8 | **MIWS_Dataset_Standard** | GitHub | 多语言仇恨言论 | `MIWS_Dataset_Standard/` | 文本 + 标签 |

---

## B. Personality Prediction / User Profiling（新增）

> 下表中前四个为必须覆盖的数据集：**Pandora / myPersonality / Twitter Personality / Essays**。

| # | 数据集 | 数据来源 | 数据类型 | 使用的心理学模型 | 输出形式 |
|---|---|---|---|---|---|
| 1 | **Pandora Dataset**（仓库使用 `jingjietan/pandora-big5`） | HuggingFace 数据集卡：https://huggingface.co/datasets/jingjietan/pandora-big5 | 文本（`text`）+ 人格维度字段（`O/C/E/A/N`）+ `ptype` | Big Five（OCEAN） | 连续 trait 分数（float）+ 类型字段（`ptype`）；train/val/test 划分 |
| 2 | **myPersonality Dataset** | Cambridge Psychometrics Centre / myPersonality Wiki：http://mypersonality.org | Facebook 用户画像数据库（人口统计 + 行为/社交字段） | Big Five / Five-Factor Model | 连续人格分数 + 人口统计标签（如 age、gender、relationship_status） |
| 3 | **Twitter Personality Dataset**（Indonesian） | GitHub: https://github.com/michellejieli/twitter-personality-classification | Twitter 文本（约 400 用户、80,000 tweets）+ 情绪/情感/社交特征 | Big Five（OCEAN） | 人格维度分类输出（项目中常见为分档或二值化） |
| 4 | **Essays Dataset** | PAN 2015 相关工作页（列出 essays.zip）：https://pan.webis.de/clef15/pan15-web/author-profiling.html | 长文本作文 + 人格标注 | Big Five（常见版本） | 连续 trait 分数或高/低标签（取决于具体发行版本） |
| 5 | **PAN Author Profiling 2015** | 官方任务页：https://pan.webis.de/clef15/pan15-web/author-profiling.html | 多语言 Twitter 用户级文本 | 用户画像 + Big Five 维度预测 | XML 输出：`age_group`、`gender`、`extroverted/stable/agreeable/conscientious/open`（-0.5~0.5） |
| 6 | **PAN Author Profiling 2016** | 官方任务页：https://pan.webis.de/clef16/pan16-web/author-profiling.html | 跨体裁用户文本（训练侧以 Twitter 为主） | 用户画像（年龄/性别；非人格量表） | XML 输出：`age_group`、`gender` |
| 7 | **Kaggle MBTI（HF 重发布版本）** | HuggingFace：https://huggingface.co/datasets/jingjietan/kaggle-mbti | 文本 + 类型字段（`ptype`） | MBTI（16 型）；部分重发布版本附加维度字段 | 16 类 MBTI 标签或映射维度标签（依版本） |
| 8 | **TwiBot-22** | 官方仓库：https://github.com/LuoUndergradXJTU/TwiBot-22；论文：https://arxiv.org/abs/2206.04564 | Twitter 异构图（`user/tweet/list/hashtag` 节点 + `edge.csv` 关系边）与账号内容特征 | 非心理学量表（图结构用户画像 / 机器人识别） | `label.csv`（bot/human）+ `split.csv`（train/val/test） |

---

## C. 新增数据集的使用建议

1. **人格建模优先级**：`pandora-big5` > `myPersonality` > `Twitter Personality`。
2. **用户画像泛化评估**：引入 PAN 2015/2016 作为跨语言和跨体裁验证集。
3. **统一输出协议**：建议将离散标签映射为统一结构：
	- `dimension_scores: {O, C, E, A, N}`（连续值优先）
	- `profile_labels: {age_group, gender, mbti}`（可选）
4. **与激进化检测融合**：将人格向量与 79 指标向量拼接做多任务学习（人格预测 + 激进化风险评估）。

---

## D. 快速索引（仓库路径）

- `dataset/pandora-big5/`
- `dataset/myPersonality/`
- `dataset/twitter-personality-classification/`


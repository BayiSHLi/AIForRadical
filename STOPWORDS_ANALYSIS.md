# 词频统计中的噪音词汇分析与改进

## 问题诊断

之前的词频统计结果中出现了不应该的词汇。问题根源分析如下：

### 1. **`rt` (2179次)** - Twitter Retweet标记
- **原因**: 不在标准NLTK stopwords中，是Twitter特定的缩写
- **解决**: 添加到自定义stopwords列表
- **改进后**: ✓ 被过滤

### 2. **`l` (1190次)** - 单个字母
- **原因**: 分词时的错误或文本中存在单独的字母，简单正则移除标点不够
- **解决**: 添加`min_word_length=2`参数，过滤所有长度<2的词汇
- **改进后**: ✓ 被过滤

### 3. **`0000` (1168次)** - 纯数字
- **原因**: 数字未被过滤，正则`[^\w\s]`保留了数字
- **解决**: 添加`remove_numbers=True`参数，检测`word.isdigit()`
- **改进后**: ✓ 被过滤

### 4. **`nov` (889次)** - 月份缩写
- **原因**: 月份缩写不在标准英文stopwords中
- **解决**: 添加月份缩写到自定义stopwords
- **改进后**: ✓ 被过滤

### 5. **`amp` (456次)** - HTML实体残留
- **原因**: `&amp;` (HTML中的&符号) 中的标点被移除后留下`amp`
- **解决**: 添加HTML实体清洗: `re.sub(r'&[a-z]+;', '', text)`
- **改进后**: ✓ 被过滤

### 6. **`itsljinny` (655次)** - Twitter Handle混入
- **原因**: 用户名被误认为是内容词汇，需要识别和移除
- **解决**: 添加mentions移除: `re.sub(r'@\w+', '', text)`
- **改进后**: ✓ 被过滤

## 改进方案

### 新增函数

#### 1. **增强型 `get_stopwords()`**
```python
def get_stopwords(use_nltk=True):
    """获取英文stopwords + Twitter特定词汇"""
    # 基础stopwords
    # + Twitter术语: 'rt', 'amp', 'nbsp'
    # + 月份: 'jan', 'feb', ..., 'nov', 'dec'
    # + 日期后缀: 'st', 'nd', 'rd', 'th'
    # + 网络术语: 'http', 'https', 'www'
    # + 单个字母: 'a-z'
```

#### 2. **高级文本清洗 `clean_text_for_analysis()`**
```python
def clean_text_for_analysis(text, stopwords_set=None, 
                            min_word_length=2, 
                            remove_numbers=True):
    """
    清洗步骤：
    1. 移除HTML实体 (&amp; → '')
    2. 移除URLs (http://...)
    3. 移除@mentions (@user → '')
    4. 移除#符号但保留单词
    5. 移除标点符号
    6. 过滤短词 (length < min_word_length)
    7. 过滤数字 (0000, 123等)
    8. 过滤stopwords
    """
```

### 新增参数

在 `analyze_content_statistics()` 中添加：
- **`remove_stopwords=True`** - 移除常见词汇（默认开启）
- **`remove_numbers=True`** - 移除纯数字（默认开启）
- **`min_word_length=2`** - 最小词长（默认2，移除单个字母）

## 使用示例

### 基础用法（推荐）
```python
dataset.analyze_content_statistics(
    top_n=30, 
    save_dir='./analysis_results',
    remove_stopwords=True,   # 移除stopwords
    remove_numbers=True,      # 移除数字
    min_word_length=2        # 移除单字母
)
```

### 仅移除stopwords，保留数字
```python
dataset.analyze_content_statistics(
    top_n=30,
    remove_stopwords=True,
    remove_numbers=False,
    min_word_length=1
)
```

### 不进行任何过滤（原始分析）
```python
dataset.analyze_content_statistics(
    top_n=30,
    remove_stopwords=False,
    remove_numbers=False,
    min_word_length=1
)
```

## 预期效果对比

### 改进前（有噪音）
```
Rank | Word           | Frequency
  1  | rt             |  2179      ← 噪音
  2  | l              |  1190      ← 噪音
  3  | jinny          |  1190      
  4  | 0000           |  1168      ← 噪音
  5  | nov            |   889      ← 噪音
  6  | allah          |   888
  ...
 22  | amp            |   456      ← 噪音
```

### 改进后（清洁）
```
Rank | Word           | Frequency
  1  | allah          |   888      ✓
  2  | new            |   848      ✓
  3  | one            |   767      ✓
  4  | like           |   765      ✓
  5  | dont           |   717      ✓
  6  | us             |   699      ✓
  7  | people         |   693      ✓
  8  | bless          |   663      ✓
  9  | video          |   600      ✓
 10  | get            |   551      ✓
```

## NLP研究中的标准实践

| 实践 | 常见度 | 推荐 | 说明 |
|------|--------|------|------|
| 移除stopwords | 极常见 | ✓✓✓ | 所有文本分析的基础 |
| 过滤短词 | 常见 | ✓✓ | 避免分词错误和噪音 |
| 移除数字 | 普遍 | ✓✓ | 通常不含语义信息 |
| 移除URL | 常见 | ✓ | 特别是社交媒体数据 |
| 移除HTML实体 | 常见 | ✓ | 网页爬取数据必需 |
| 移除特殊符号 | 通用 | ✓✓✓ | 标准预处理步骤 |
| 小写转换 | 通用 | ✓✓✓ | 避免大小写重复 |
| 去除重音符号 | 常见 | ✓ | 多语言处理 |
| 词干提取/词根化 | 常见 | ✓ | 高级NLP分析 |

## 技术细节

### 正则表达式用法
```python
# HTML实体
re.sub(r'&[a-z]+;', '', text)     # &amp; &nbsp; &lt; 等

# URLs
re.sub(r'http[s]?://\S+', '', text)

# @mentions (Twitter/社交媒体)
re.sub(r'@\w+', '', text)

# #hashtags (移除#但保留单词)
re.sub(r'#', '', text)

# 标点符号 (保留内部撇号)
re.sub(r"[^\w\s']", '', text)

# 规范化空格
re.sub(r"\s+", ' ', text)
```

### 词长度过滤的含义
- `min_word_length=1`: 保留所有词（包括单个字母）
- `min_word_length=2`: **推荐** - 移除噪音但保留有意义词汇
- `min_word_length=3`: 更严格 - 适合某些特殊分析

## 参考资源

- **NLTK Stopwords**: https://www.nltk.org/howto/wordnet.html
- **Twitter文本处理**: Tweet tokenizer, porter stemmer
- **NLP最佳实践**: ACL (Association for Computational Linguistics) 会议论文

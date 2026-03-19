import re
import json
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex

# ===== Embedding（显式）=====
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# ===== 读取codebook.txt并解析indicators =====
BASE_DIR = Path(__file__).resolve().parent
CODEBOOK_PATH = BASE_DIR / "codebook.txt"

with open(CODEBOOK_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# ===== 使用正则表达式提取所有indicator =====
# 模式: [编号] 分类1: 分类2 >> 指标名
pattern = r"\[(\d+)\]\s+(.+?)\s+>>\s+(.+?)(?=\n\-+\n|$)"
indicator_matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

documents = []
indicator_list = []

for match in indicator_matches:
    number = match.group(1)
    category = match.group(2).strip()
    indicator_name = match.group(3).strip()
    
    # 完整的indicator标识符格式
    full_indicator = f"[{number}] {category} >> {indicator_name}"
    
    # 提取该indicator后面的样本部分（Top Sample）
    indicator_start = match.start()
    # 查找下一个indicator或文件末尾
    next_match = None
    for m in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        if m.start() > indicator_start:
            next_match = m
            break
    
    if next_match:
        indicator_content = content[indicator_start:next_match.start()]
    else:
        indicator_content = content[indicator_start:]
    
    # 从内容中提取Top Sample
    samples = []
    sample_pattern = r"Top Sample #(\d+):(.*?)(?=\n\s*Top Sample #|\n\s*[-]{10,}|\Z)"
    sample_matches = re.finditer(sample_pattern, indicator_content, re.DOTALL)
    
    for sample_match in sample_matches:
        sample_text = sample_match.group(2).strip()
        # 只取前200字符作为简洁版本
        sample_summary = sample_text[:300] if len(sample_text) > 300 else sample_text
        samples.append(sample_summary)
    
    # 如果没有样本，标记为"未找到匹配"
    if not samples:
        samples = ["(No matching samples found with sufficient similarity)"]
    
    # 构建文档文本
    samples_text = "\n\n".join([f"Example {i+1}:\n{s}" for i, s in enumerate(samples[:3])])
    
    text = f"""INDICATOR ID: {full_indicator}

CATEGORY: {category}
SHORT NAME: {indicator_name}

DEFINITION:
This indicator represents {indicator_name} within the context of {category}.

PROTOTYPE EXAMPLES:
{samples_text}

NOTE: When detecting this indicator in social media posts, look for:
- Themes related to {indicator_name}
- Language patterns from known content
- Contextual alignment with the category {category}
"""
    
    documents.append(Document(text=text, metadata={"indicator_id": full_indicator, "short_name": indicator_name}))
    indicator_list.append(full_indicator)

print(f"✓ Extracted {len(indicator_list)} indicators from codebook")
print(f"  First 5 indicators:")
for ind in indicator_list[:5]:
    print(f"    - {ind}")

# ===== Document → Nodes =====
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# ===== StorageContext =====
storage_context = StorageContext.from_defaults()

# ===== 构建 Index =====
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

index.storage_context.persist(str(BASE_DIR / "rule_index"))

# ===== 保存indicator mapping供detect.py使用 =====
mapping_file = BASE_DIR / "indicator_mapping.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_indicators": len(indicator_list),
        "indicators": indicator_list
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Rule index rebuilt successfully")
print(f"✓ Indicator mapping saved to {mapping_file}")
print(f"✓ Total indicators: {len(indicator_list)}")

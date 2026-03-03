#!/usr/bin/env python3
"""Test script to verify indicator mapping and prompt format"""

import os
import json
import re
import sys

# 测试 build_rule_nodes.py 的指标解析逻辑
print("="*60)
print("TESTING INDICATOR EXTRACTION")
print("="*60)

codebook_path = "/home/user/workspace/SHLi/AI for radicalisation/RAGBasedAI/codebook.txt"

if not os.path.exists(codebook_path):
    print(f"❌ Codebook not found: {codebook_path}")
    sys.exit(1)

with open(codebook_path, "r") as f:
    content = f.read()

# 使用与 build_rule_nodes.py 相同的正则表达式
pattern = r"\[(\d+)\]\s+(.+?)\s+>>\s+(.+?)(?=\n\-+\n|$)"
indicator_matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

indicators = []
for i, match in enumerate(indicator_matches):
    number = match.group(1)
    category = match.group(2).strip()
    indicator_name = match.group(3).strip()
    
    full_indicator = f"[{number}] {category} >> {indicator_name}"
    indicators.append(full_indicator)
    
    if i < 5:
        print(f"\n✓ Indicator {i+1}:")
        print(f"  Format: {full_indicator}")
        print(f"  Number: {number}")
        print(f"  Category: {category}")
        print(f"  Name: {indicator_name}")

print(f"\n{'='*60}")
print(f"✓ Total indicators extracted: {len(indicators)}")
print(f"{'='*60}")

# 显示所有指标
print("\nAll Indicators:")
for ind in indicators:
    print(f"  {ind}")

# 模拟 LLM 输出格式
print(f"\n{'='*60}")
print("EXAMPLE LLM OUTPUT FORMAT")
print(f"{'='*60}")

example_output = {
    "has_radicalisation_indicators": True,
    "indicators_detected": [
        "[1] Need: Individual Loss >> individual_loss_interpersonal",
        "[29] Narrative: Violent >> narrative_violent_salafi_jihadism"
    ],
    "reasoning": "The post exhibits themes related to individual loss and salafi jihadist narratives",
    "confidence_level": "medium",
    "evidence_references": "Similar to documented content patterns in indicators [1] and [29]"
}

print("\nExpected JSON output:")
print(json.dumps(example_output, indent=2, ensure_ascii=False))

print(f"\n{'='*60}")
print("✅ Indicator mapping test complete")
print(f"{'='*60}")

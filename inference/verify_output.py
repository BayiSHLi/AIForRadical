"""
验证脚本：检查 LLM 输出是否仅包含新生成的内容
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
torch.set_grad_enabled(False)

# 简化的 system prompt
SYSTEM_PROMPT = """You are a coder. Respond ONLY with valid JSON.
{
  "indicators": {
    "indicator1": "Present",
    "indicator2": "Not Present"
  }
}"""

# 测试内容
test_content = "This is about jihad and religious activities."

# 构建消息
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Content: {test_content}\nRespond with JSON format."}
]

# 应用 chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 分词
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
).to(DEVICE)

input_length = inputs['input_ids'].shape[-1]

print(f"\n{'='*80}")
print("INPUT ANALYSIS")
print(f"{'='*80}")
print(f"Input token length: {input_length}")
print(f"First 20 tokens of input: {inputs['input_ids'][0][:20].tolist()}")

# 生成
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

total_length = output.shape[-1]
print(f"Total output token length: {total_length}")
print(f"New tokens generated: {total_length - input_length}")

# 方式 A：错误的解码方式（包含输入）
print(f"\n{'='*80}")
print("❌ WRONG: Decode entire sequence (包含 system prompt)")
print(f"{'='*80}")
full_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Length: {len(full_decoded)} characters")
print(f"Preview: {full_decoded[:200]}...\n")
print(f"Contains 'You are a coder': {'You are a coder' in full_decoded}")
print(f"Contains 'system prompt': {'system prompt' in full_decoded or 'coder' in full_decoded}")

# 方式 B：正确的解码方式（仅新生成部分）
print(f"\n{'='*80}")
print("✅ CORRECT: Decode only new tokens (不包含 system prompt)")
print(f"{'='*80}")
new_tokens = output[0][input_length:]
decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"Length: {len(decoded)} characters")
print(f"Content: {decoded}\n")
print(f"Contains 'You are a coder': {'You are a coder' in decoded}")
print(f"Is valid JSON structure: {decoded.strip().startswith('{')}")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print(f"✓ 使用方式 B（仅解码新生成的token）")
print(f"✓ 已在 llm_inference_fast.py 中修复")
print(f"✓ 输出应该仅包含 JSON 格式的指标编码结果")
print(f"{'='*80}")

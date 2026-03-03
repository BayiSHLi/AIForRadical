"""
批处理效果验证脚本
对比逐个推理 vs 真正的批处理性能差异
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
MAX_LENGTH = 3072

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
torch.set_grad_enabled(False)

SYSTEM_PROMPT = "You are a coder. Respond with JSON format."

# 生成测试数据
test_samples = [
    {"content": f"Sample {i}: This is about jihad and religious activities. {i*'test '}", 
     "person": f"Person_{i}", 
     "category": f"Category_{i}", 
     "date": f"2020-01-{i:02d}"}
    for i in range(1, BATCH_SIZE + 1)
]

print(f"\n{'='*80}")
print(f"BATCH PROCESSING TEST: {BATCH_SIZE} samples")
print(f"{'='*80}\n")

# ============================================================================
# 方式 A：逐个推理（旧方法 - 低效）
# ============================================================================
print("❌ 方式 A：逐个推理（sequential）")
print("-" * 80)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.time()

outputs_sequential = []
for item in test_samples:
    user_prompt = f"Content: {item['content']}\nRespond with JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    input_length = inputs['input_ids'].shape[-1]
    new_tokens = output[0][input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    outputs_sequential.append(decoded)

torch.cuda.synchronize()
sequential_time = time.time() - start_time
sequential_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f"Time: {sequential_time:.2f}s ({sequential_time/BATCH_SIZE:.3f}s/sample)")
print(f"Peak Memory: {sequential_memory:.2f} GB")
print(f"Output samples: {len(outputs_sequential)}\n")

# ============================================================================
# 方式 B：真正的批处理（新方法 - 高效）
# ============================================================================
print("✅ 方式 B：批处理（batch）")
print("-" * 80)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.time()

# 步骤1：批量构建提示
prompts = []
for item in test_samples:
    user_prompt = f"Content: {item['content']}\nRespond with JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompts.append(prompt)

# 步骤2：批量分词（带padding）
tokenized = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH
).to(DEVICE)

# 记录每个样本的实际输入长度
input_lengths = []
for i in range(len(test_samples)):
    input_ids = tokenized['input_ids'][i]
    actual_length = (input_ids != tokenizer.pad_token_id).sum().item()
    input_lengths.append(actual_length)

# 步骤3：批量生成
with torch.no_grad():
    all_outputs = model.generate(
        **tokenized,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

# 步骤4：批量解码
outputs_batch = []
for i, output in enumerate(all_outputs):
    input_length = input_lengths[i]
    new_tokens = output[input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    outputs_batch.append(decoded)

torch.cuda.synchronize()
batch_time = time.time() - start_time
batch_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f"Time: {batch_time:.2f}s ({batch_time/BATCH_SIZE:.3f}s/sample)")
print(f"Peak Memory: {batch_memory:.2f} GB")
print(f"Output samples: {len(outputs_batch)}\n")

# ============================================================================
# 性能对比
# ============================================================================
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")
speedup = sequential_time / batch_time
memory_increase = (batch_memory / sequential_memory - 1) * 100

print(f"Speed improvement:   {speedup:.2f}x faster")
print(f"Time per sample:     {sequential_time/BATCH_SIZE:.3f}s → {batch_time/BATCH_SIZE:.3f}s")
print(f"Memory usage:        {sequential_memory:.2f} GB → {batch_memory:.2f} GB ({memory_increase:+.1f}%)")
print(f"\nExpected total time (18000 samples):")
print(f"  Sequential: {18000 * (sequential_time/BATCH_SIZE) / 3600:.1f} hours")
print(f"  Batch:      {18000 * (batch_time/BATCH_SIZE) / 3600:.1f} hours")

print(f"\n{'='*80}")
if speedup > 1.5:
    print(f"✅ 批处理有效！已实现 {speedup:.1f}x 加速")
else:
    print(f"⚠️  批处理效果不明显，可能的原因：")
    print(f"   - 样本大小差异大导致padding浪费")
    print(f"   - GPU 显存足够，批处理收益有限")
    print(f"   - 建议尝试更大的 BATCH_SIZE")
print(f"{'='*80}\n")

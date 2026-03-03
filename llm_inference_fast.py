import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import RadicalisationDataset
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import threading
from queue import Queue

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ============================
# CONFIGURATION
# ============================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE_IDS = [0, 1]  # 使用两块显卡
PRIMARY_DEVICE = "cuda:0"

# 推理参数
BATCH_SIZE = 2  # 批处理大小（A6000 48G可以轻松支持）
MAX_LENGTH = 1024  # 增大总长度以容纳长输出
MAX_NEW_TOKENS = 1024  # 足以覆盖51个指标的JSON输出（需要1300-1500 tokens）

# 输出文件
OUTPUT_JSONL = "llm_results_fast.jsonl"
OUTPUT_STATS = "llm_inference_stats_fast.json"
CHECKPOINT_INTERVAL = 50  # 每50样本保存一次

# ============================
# 多GPU 模型加载
# ============================
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using devices: {DEVICE_IDS}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配到多个GPU
)

# 设置为评估模式并禁用梯度
model.eval()
torch.set_grad_enabled(False)

print(f"✓ Model loaded\n")

# ============================
# SYSTEM PROMPT (预编译)
# ============================
SYSTEM_PROMPT = """You are acting as a trained human coder following the indicator-based coding framework proposed in Neo (2020) for detecting markers of radicalisation in social media text.

Your task is NOT to assess whether the author is radicalised.
Your task is to identify whether specific indicators are PRESENT or NOT PRESENT in the given text.

You must follow these rules strictly:
1. Base your judgement ONLY on the explicit content of the given text.
2. Do NOT infer intent, background, or offline behaviour.
3. Each indicator must be coded independently.
4. Use binary coding: "Present" or "Not Present".
5. An indicator is "Present" only if there is clear linguistic evidence in the text.

Additional constraints:
- The fields "Person", "Category", and "Date" are provided for contextual completeness only.
- You MUST NOT use the identity of the person, group labels, or metadata to make your judgement.
- Coding decisions must be based solely on the textual content labelled as "Content".

Below is the list of indicators you must code. You must evaluate ALL indicators for each input text.

--------------------------------------------------
NEED
Need: Individual Significance Loss: Interpersonal  
Need: Individual Significance Loss: Career  
Need: Individual Significance Loss: Religious  
Need: Individual Significance Loss: Radical Activities  
Need: Individual Significance Loss: Health  
Need: Individual Significance Loss: Finances  
Need: Individual Significance Loss: Education  
Need: Individual Significance Loss: Self-esteem  
Need: Individual Significance Loss: Others  
Need: Social Significance Loss: Radical Religious  
Need: Social Significance Loss: Non-radical Religious  
Need: Social Significance Loss: Non-religious  
Need: Opportunity for Significance Gain: Leadership  
Need: Opportunity for Significance Gain: Martyrdom  
Need: Opportunity for Significance Gain: Vengeance  
Need: Opportunity for Significance Gain: Career  
Need: Opportunity for Significance Gain: Interpersonal  
Need: Opportunity for Significance Gain: Religious  
Need: Opportunity for Significance Gain: Educational  
Need: Opportunity for Significance Gain: Training  
Need: Opportunity for Significance Gain: Radical Activities  
Need: Opportunity for Significance Gain: Miscellaneous  
Need: Quest for Significance: Radical  
Need: Quest for Significance: Non-radical  
Need: Quest for Significance: Dualistic  
Need: Quest for Significance: Competing  

--------------------------------------------------
NARRATIVE
Narrative: Violent: Necessity  
Narrative: Violent: Allowability  
Narrative: Violent: Salafi Jihadism  
Narrative: Violent: Takfiri  
Narrative: Violent: Jihad qital  
Narrative: Violent: Martyrdom  
Narrative: Non-Violent: Thogut  
Narrative: Non-Violent: Baiat  
Narrative: Non-Violent: Muslim Brotherhood  
Narrative: Non-Violent: Salafi  
Narrative: Non-Violent: Jihad  
Narrative: Non-Violent: Rida  
Narrative: Non-Violent: Political Views  
Narrative: Disagreement with Radical Group(s): Unspecified  
Narrative: Disagreement with Radical Group(s): Military/violent aspects  
Narrative: Disagreement with Radical Group(s): Political aspects  
Narrative: Disagreement with Radical Group(s): Strategies/Methods  
Narrative: Disagreement with Radical Group(s): Religious aspects  
Narrative: Disagreement with Radical Ideology: Takfiri  
Narrative: Disagreement with Radical Ideology: Salafi  
Narrative: Disagreement with Radical Ideology: Thogut  
Narrative: Religious Historical References  
Narrative: Differences among Radical Groups  
Narrative: Unspecified  

--------------------------------------------------

Output format:
{
  "Need": {"<indicator>": "Present / Not Present", ...},
  "Narrative": {"<indicator>": "Present / Not Present", ...}
}"""

def make_user_prompt(person, category, date, content):
    """构建用户提示"""
    return f"""Contextual information (do NOT use this information for coding):
- Person: {person}
- Category: {category}
- Date: {date}

Text content to be coded:
-------------------------
{content}
-------------------------

Return the result using the predefined JSON format."""

# ============================
# 辅助函数
# ============================
def load_processed_ids():
    """加载已处理样本的ID"""
    processed = set()
    if os.path.exists(OUTPUT_JSONL):
        try:
            with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed.add(data.get("id"))
        except Exception as e:
            print(f"⚠️  Warning: {e}")
    return processed

def batch_inference(batch_data: List[Dict]) -> List[str]:
    """
    真正的批处理推理 - 一次性处理多个样本
    
    Args:
        batch_data: List of {'content', 'person', 'category', 'date'}
    
    Returns:
        List of LLM outputs
    """
    outputs = []
    
    # 关键优化：构建所有提示，但只用 system prompt 一次
    prompts = []
    input_lengths = []
    

    print(f"    [Step 1] Building prompts for {len(batch_data)} samples...")

    with torch.no_grad():
        for item in batch_data:
            user_prompt = make_user_prompt(
                item['person'], item['category'], item['date'], item['content']
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            # 格式化提示
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # 步骤1：批量分词（关键优化点）
        # tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # 自动填充到相同长度
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # 记录输入长度（用于后续提取新生成的token）
        for i in range(len(batch_data)):
            # 计算每个样本的实际输入长度（去除padding）
            input_ids = tokenized['input_ids'][i]
            actual_length = (input_ids != tokenizer.pad_token_id).sum().item()
            input_lengths.append(actual_length)
        
        # 步骤2：批量生成（真正的批处理）
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": 0.2,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # 一次性生成所有样本
        all_outputs = model.generate(
            input_ids=tokenized["input_ids"], 
            attention_mask=tokenized["attention_mask"], 
            **gen_kwargs)
        
        # 步骤3：批量解码（仅新生成的部分）
        for i, output in enumerate(all_outputs):
            input_length = input_lengths[i]
            new_tokens = output[input_length:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(decoded)
    
    return outputs

# ============================
# 异步 I/O 线程
# ============================
class JSONLWriter:
    """异步写入JSONL的线程"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = Queue(maxsize=100)
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
    
    def _writer_loop(self):
        """后台线程：持续从队列读取并写入"""
        with open(self.filepath, 'a', encoding='utf-8', buffering=1024*1024) as f:
            while True:
                item = self.queue.get()
                if item is None:  # 停止信号
                    break
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                self.queue.task_done()
    
    def write(self, data):
        """非阻塞写入"""
        self.queue.put(data)
    
    def flush(self):
        """等待队列清空"""
        self.queue.join()
    
    def close(self):
        """关闭写入器"""
        self.queue.put(None)
        self.thread.join()

# ============================
# 主推理循环
# ============================
# ROOT_DIR = r"c:\Users\shanghong.li\Desktop\AI for radicalisation\Fighter and sympathiser"
# for workstation
ROOT_DIR = r"/home/user/workspace/SHLi/AI for radicalisation/Fighter and sympathiser"
dataset = RadicalisationDataset(ROOT_DIR)

processed_ids = load_processed_ids()
already_done = len(processed_ids)
total_samples = len(dataset.data)
remaining = total_samples - already_done

print("\n" + "="*80)
print("LLM Inference - BATCH OPTIMIZED VERSION")
print("="*80)
print(f"Total samples: {total_samples}")
print(f"Already processed: {already_done}")
print(f"Remaining: {remaining}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {PRIMARY_DEVICE} (+ auto device_map)")
print(f"Output: {OUTPUT_JSONL}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

if remaining == 0:
    print("✓ All samples already processed!")
else:
    start_time = time.time()
    processed_count = 0
    error_count = 0
    errors_list = []
    
    # 启动异步写入线程
    writer = JSONLWriter(OUTPUT_JSONL)
    
    try:
        batch_buffer = []
        batch_indices = []
        
        for idx, raw_row in enumerate(dataset.data):
            if idx in processed_ids:
                continue
            
            try:
                # 提取行数据
                if isinstance(raw_row, pd.Series):
                    row_dict = raw_row.to_dict()
                else:
                    row_dict = dict(raw_row)
                
                person = str(row_dict.get("person", ""))
                category = str(row_dict.get("category", ""))
                date = str(row_dict.get("date", ""))
                content = str(row_dict.get("content", ""))
                gold_coded = row_dict.get("coded", "")
                
                # 添加到批处理缓冲
                batch_buffer.append({
                    'content': content,
                    'person': person,
                    'category': category,
                    'date': date,
                })
                batch_indices.append((idx, person, category, date, gold_coded, row_dict))
                
                # 当缓冲达到批大小或到达数据集末尾时执行推理
                if len(batch_buffer) >= BATCH_SIZE or (idx == total_samples - 1 and batch_buffer):
                    # 批处理推理
                    print(f"\n[Batch] Processing {len(batch_buffer)} samples...")
                    start_batch = time.time()
                    outputs = batch_inference(batch_buffer)
                    batch_time = time.time() - start_batch
                    
                    # 保存结果
                    for (sample_idx, p, c, d, gc, row_d), output in zip(batch_indices, outputs):
                        result = {
                            "id": sample_idx,
                            "person": p,
                            "category": c,
                            "date": d,
                            "gold_coded": gc,
                            "llm_output": output,
                            "inference_time_sec": round(batch_time / len(batch_buffer), 2)
                        }
                        writer.write(result)
                        processed_count += 1
                        
                        # 进度显示
                        progress = already_done + processed_count
                        elapsed = time.time() - start_time
                        avg_time = elapsed / processed_count
                        remaining_samples = remaining - processed_count
                        eta_seconds = avg_time * remaining_samples
                        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                        
                        pct = (progress / total_samples * 100)
                        print(f"  [{progress}/{total_samples}] {p} | {pct:.1f}% | ETA: {eta_time.strftime('%H:%M:%S')}")
                    
                    print(f"  ✓ Batch done in {batch_time:.2f}s ({batch_time/len(batch_buffer):.2f}s/sample)\n")
                    
                    # 清空缓冲
                    batch_buffer = []
                    batch_indices = []
                    
                    # 定期保存检查点
                    if processed_count % CHECKPOINT_INTERVAL == 0:
                        writer.flush()
                        print(f"  💾 Checkpoint: {processed_count} samples saved\n")
            
            except Exception as e:
                error_count += 1
                error_msg = str(e)[:100]
                print(f"  ✗ Error at sample {idx}: {error_msg}\n")
                errors_list.append({"sample_id": idx, "person": person, "error": error_msg})
                
                result = {
                    "id": idx,
                    "person": person,
                    "category": category,
                    "date": date,
                    "gold_coded": gold_coded,
                    "llm_output": f"ERROR: {error_msg}",
                    "inference_time_sec": 0,
                    "error": True
                }
                writer.write(result)
                processed_count += 1
    
    finally:
        # 关闭异步写入器
        writer.close()
    
    # 最终统计
    total_time = time.time() - start_time
    final_progress = already_done + processed_count
    
    print("\n" + "="*80)
    print("Inference Complete")
    print("="*80)
    print(f"This run: {processed_count} samples")
    print(f"Successful: {processed_count - error_count}")
    print(f"Errors: {error_count}")
    print(f"Overall progress: {final_progress}/{total_samples}")
    print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
    print(f"Avg/sample: {total_time/processed_count:.3f}s")
    print(f"Throughput: {processed_count/total_time:.1f} samples/sec")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {OUTPUT_JSONL}")
    
    if error_count > 0:
        print(f"\n⚠️  Errors ({error_count}):")
        for err in errors_list[:5]:
            print(f"  - ID {err['sample_id']}: {err['error']}")
        if error_count > 5:
            print(f"  ... and {error_count - 5} more")
    
    # 保存统计信息
    stats = {
        "model": MODEL_NAME,
        "device": PRIMARY_DEVICE,
        "batch_size": BATCH_SIZE,
        "total_samples": total_samples,
        "processed": final_progress,
        "successful": final_progress - error_count,
        "errors": error_count,
        "time_sec": round(total_time, 2),
        "avg_time_per_sample": round(total_time / processed_count, 3),
        "throughput_per_sec": round(processed_count / total_time, 2),
        "timestamp": datetime.now().isoformat()
    }
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats: {OUTPUT_STATS}")
    
    print("="*80)
    
    # 如果全部完成，转换为CSV
    if final_progress == total_samples:
        print("\n📊 Converting to CSV...")
        try:
            rows = []
            with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
            df.to_csv("llm_results_fast.csv", index=False)
            print(f"✓ CSV export: llm_results_fast.csv")
        except Exception as e:
            print(f"⚠️  Could not export CSV: {e}")

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import RadicalisationDataset


def resolve_dataset_root() -> Path:
    """Resolve dataset directory across possible repository layouts."""
    candidates = [
        PROJECT_ROOT / "data" / "Fighter and sympathiser",
        PROJECT_ROOT / "Fighter and sympathiser",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    checked = "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(
        "Dataset root not found. Checked:\n"
        f"{checked}"
    )

# ============================
# CONFIGURATION
# ============================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output files
OUTPUT_JSONL = "llm_results.jsonl"
OUTPUT_STATS = "llm_inference_stats.json"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N samples

# Load model and tokenizer once
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
print(f"✓ Model loaded on {DEVICE}\n")

# Disable gradient computation (memory optimization)
torch.set_grad_enabled(False)

# ============================
# PROMPTS (Pre-compiled)
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

Below is the list of indicators you must code.
You must evaluate ALL indicators for each input text.

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

For each indicator, output:
- Indicator name
- Coding result: Present or Not Present

Do not provide explanations unless explicitly requested.
Do not merge indicators.
Do not add new categories.

Your output must strictly follow the required format.

{
  "Need": {
    "<indicator name>": "Present / Not Present",
    ...
  },
  "Narrative": {
    "<indicator name>": "Present / Not Present",
    ...
  }
}"""

def make_user_prompt(person, category, date, content):
    """Build user prompt (avoid repeated string formatting)"""
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
# HELPER FUNCTIONS
# ============================
def load_processed_ids():
    """Load already-processed sample IDs from JSONL"""
    processed = set()
    if os.path.exists(OUTPUT_JSONL):
        try:
            with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed.add(data.get("id"))
        except Exception as e:
            print(f"⚠️  Warning: Could not read checkpoint: {e}")
    return processed

def append_result_jsonl(result, file_handle=None):
    """Append single result to JSONL (pass file handle for batch writes)"""
    if file_handle is None:
        # Single write mode (backward compat)
        with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        # Batch mode: write to provided handle
        file_handle.write(json.dumps(result, ensure_ascii=False) + '\n')

def run_llm_inference(content, person, category, date, max_new_tokens=1024):
    """Optimized inference: minimal conversions, compiled prompts"""
    user_prompt = make_user_prompt(person, category, date, content)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def convert_jsonl_to_csv(jsonl_path, csv_path):
    """Convert JSONL to CSV for compatibility"""
    try:
        rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV export: {csv_path}")
    except Exception as e:
        print(f"⚠️  Could not export CSV: {e}")

# ============================
# MAIN INFERENCE LOOP
# ============================
ROOT_DIR = str(resolve_dataset_root())
dataset = RadicalisationDataset(ROOT_DIR)

# Load checkpoint
processed_ids = load_processed_ids()
already_done = len(processed_ids)
total_samples = len(dataset.data)
remaining = total_samples - already_done

print("\n" + "="*80)
print("LLM Inference with Checkpoint Support")
print("="*80)
print(f"Total samples: {total_samples}")
print(f"Already processed: {already_done}")
print(f"Remaining: {remaining}")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_JSONL}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

if remaining == 0:
    print("✓ All samples already processed!")
    print(f"Results: {OUTPUT_JSONL}")
else:
    start_time = time.time()
    processed_count = 0
    error_count = 0
    errors_list = []

    # Open file handle once (not per sample!)
    jsonl_file = open(OUTPUT_JSONL, 'a', encoding='utf-8', buffering=1024*1024)  # 1MB buffer

    for idx, raw_row in enumerate(dataset.data):
        # Skip already processed
        if idx in processed_ids:
            continue

        try:
            # Extract row (optimize type checking)
            if isinstance(raw_row, pd.Series):
                row_dict = raw_row.to_dict()
            else:
                row_dict = dict(raw_row)

            person = str(row_dict.get("person", ""))
            category = str(row_dict.get("category", ""))
            date = str(row_dict.get("date", ""))
            content = str(row_dict.get("content", ""))
            gold_coded = row_dict.get("coded", "")

            # Display progress
            content_preview = (content[:50] + "...") if len(content) > 50 else content
            progress = already_done + processed_count + 1
            print(f"[{progress}/{total_samples}] {person} | {category} | {date[:10]}")
            print(f"               {content_preview}")

            # Run inference
            start_inf = time.time()
            llm_output = run_llm_inference(content, person, category, date)
            inf_time = time.time() - start_inf

            # Append to JSONL (pass file handle for buffered write)
            result = {
                "id": idx,
                "person": person,
                "category": category,
                "date": date,
                "gold_coded": gold_coded,
                "llm_output": llm_output,
                "inference_time_sec": round(inf_time, 2)
            }
            append_result_jsonl(result, jsonl_file)
            processed_count += 1

            # ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            remaining_samples = remaining - processed_count
            eta_seconds = avg_time * remaining_samples
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)

            print(f"               ✓ {inf_time:.2f}s | {(progress/total_samples*100):.1f}% | ETA: {eta_time.strftime('%H:%M:%S')}\n")

            # Checkpoint: flush buffer every N samples
            if processed_count % CHECKPOINT_INTERVAL == 0:
                jsonl_file.flush()
                print(f"   💾 Saved {processed_count} samples\n")

        except Exception as e:
            error_count += 1
            error_msg = str(e)[:100]
            print(f"               ✗ {error_msg}\n")
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
            append_result_jsonl(result, jsonl_file)
            processed_count += 1

    # Close file handle
    jsonl_file.close()

    # Summary
    total_time = time.time() - start_time
    final_progress = already_done + processed_count

    print("\n" + "="*80)
    print("Inference Complete")
    print("="*80)
    print(f"This run: {processed_count} samples")
    print(f"Successful: {processed_count - error_count}")
    print(f"Errors: {error_count}")
    print(f"Overall progress: {final_progress}/{total_samples}")
    print(f"Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Avg/sample: {total_time/processed_count:.2f}s")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {OUTPUT_JSONL}")

    if error_count > 0:
        print(f"\n⚠️  Errors ({error_count}):")
        for err in errors_list[:5]:
            print(f"  - ID {err['sample_id']}: {err['error']}")
        if error_count > 5:
            print(f"  ... and {error_count - 5} more")

    # Save stats
    stats = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "total_samples": total_samples,
        "processed": final_progress,
        "successful": final_progress - error_count,
        "errors": error_count,
        "time_sec": round(total_time, 2),
        "avg_time_per_sample": round(total_time / processed_count, 2),
        "timestamp": datetime.now().isoformat()
    }
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats: {OUTPUT_STATS}")

    print("="*80)

    # Convert to CSV when done
    if final_progress == total_samples:
        print("\n📊 Converting to CSV...")
        convert_jsonl_to_csv(OUTPUT_JSONL, "llm_results.csv")

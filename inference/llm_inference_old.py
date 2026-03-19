import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path

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

# -----------------------------
# 1. Model config
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 可替换
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# -----------------------------
# 2. System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are acting as a trained human coder following the indicator-based coding framework proposed in Neo (2020) for detecting markers of radicalisation in social media text.

Your task is NOT to assess whether the author is radicalised.
Your task is to identify whether specific indicators are PRESENT or NOT PRESENT in the given text.

You must follow these rules strictly:
1. Base your judgement ONLY on the explicit content of the given text.
2. Do NOT infer intent, background, or offline behaviour.
3. Each indicator must be coded independently.
4. Use binary coding: “Present” or “Not Present”.
5. An indicator is “Present” only if there is clear linguistic evidence in the text.

Additional constraints:
- The fields “Person”, “Category”, and “Date” are provided for contextual completeness only.
- You MUST NOT use the identity of the person, group labels, or metadata to make your judgement.
- Coding decisions must be based solely on the textual content labelled as “Content”.

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
}


"""

# -----------------------------
# 3. User Prompt Template
# -----------------------------
USER_PROMPT_TEMPLATE = """
Contextual information (do NOT use this information for coding):
- Person: {person}
- Category: {category}
- Date: {date}

Text content to be coded:
-------------------------
{content}
-------------------------

Return the result using the predefined JSON format.
"""

# -----------------------------
# 4. Load dataset (use existing RadicalisationDataset)
# -----------------------------
ROOT_DIR = str(resolve_dataset_root())
dataset = RadicalisationDataset(ROOT_DIR)

# We'll iterate over `dataset.data`, which contains normalized rows

# -----------------------------
# 5. Inference function
# -----------------------------
def run_llm(system_prompt, user_prompt, max_new_tokens=1024):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# -----------------------------
# 6. Run on dataset
# -----------------------------
results = []
total_samples = len(dataset.data)
errors = []

print("\n" + "="*80)
print(f"Starting Inference on Dataset")
print("="*80)
print(f"Total samples: {total_samples}")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

start_time = time.time()

for idx, raw_row in enumerate(dataset.data):
    try:
        # rows may be dict or pd.Series depending on how dataset was constructed
        if isinstance(raw_row, pd.Series):
            row = raw_row.to_dict()
        else:
            row = dict(raw_row)

        person = row.get("person", "")
        category = row.get("category", "")
        date = row.get("date", "")
        content = row.get("content", "")

        # Log current sample info
        content_preview = (content[:50] + "...") if len(content) > 50 else content
        print(f"[{idx+1}/{total_samples}] Processing: Person={person}, Category={category}, Date={date}")
        print(f"            Content preview: {content_preview}")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            person=person,
            category=category,
            date=date,
            content=content
        )

        output = run_llm(SYSTEM_PROMPT, user_prompt)
        results.append({
            "id": idx,
            "llm_output": output,
            "gold_coded": row.get("coded", ""),
            "person": person,
            "category": category,
            "date": date
        })
        
        # Progress indicator
        progress_pct = ((idx + 1) / total_samples) * 100
        elapsed = time.time() - start_time
        avg_time_per_sample = elapsed / (idx + 1)
        remaining_samples = total_samples - (idx + 1)
        eta_seconds = avg_time_per_sample * remaining_samples
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        
        print(f"            ✓ Complete | Progress: {progress_pct:.1f}% | ETA: {eta_time.strftime('%H:%M:%S')}\n")
        
    except Exception as e:
        error_msg = f"Error at sample {idx}: {str(e)}"
        print(f"            ✗ ERROR: {error_msg}\n")
        errors.append({
            "sample_id": idx,
            "person": row.get("person", ""),
            "error": str(e)
        })
        results.append({
            "id": idx,
            "llm_output": f"ERROR: {str(e)}",
            "gold_coded": row.get("coded", ""),
            "person": person,
            "category": category,
            "date": date
        })

# -----------------------------
# 7. Save results and print summary
# -----------------------------
end_time = time.time()
total_time = end_time - start_time

out_df = pd.DataFrame(results)
out_df.to_csv("llm_results.csv", index=False)

print("\n" + "="*80)
print("Inference Complete")
print("="*80)
print(f"Total samples processed: {len(results)}")
print(f"Successful: {total_samples - len(errors)}")
print(f"Errors: {len(errors)}")
print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Average time per sample: {total_time/total_samples:.2f} seconds")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results saved to: llm_results.csv")

if errors:
    print(f"\n⚠️  Errors encountered:")
    for err in errors:
        print(f"  - Sample {err['sample_id']}: {err['error'][:100]}...")

print("="*80)

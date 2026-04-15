from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "holistic-ai/personality_classifier"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "personality_classifier"


def download_model(local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_dir))
    return local_dir


def predict(texts: list[str], model_dir: Path) -> list[tuple[str, dict[str, float]]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.eval()

    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label
    results = []
    for idx, text in enumerate(texts):
        row = probs[idx]
        label_scores = {
            id2label[label_id]: float(row[label_id])
            for label_id in range(row.shape[0])
        }
        best_id = int(torch.argmax(row).item())
        results.append((text, {"top_label": id2label[best_id], **label_scores}))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and test holistic-ai/personality_classifier"
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help="Local directory used to store model files",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Input text to classify (can pass multiple times)",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    print(f"Downloading model to: {local_dir}")
    download_model(local_dir)

    texts = args.text or [
        "I enjoy planning things carefully and keeping routines.",
        "I love trying new experiences and exploring unusual ideas.",
        "I get stressed quickly when too many things happen at once.",
    ]

    print(f"\nLoaded {len(texts)} test text(s). Running inference...\n")
    results = predict(texts, local_dir)

    for i, (text, scores) in enumerate(results, start=1):
        print(f"[{i}] text: {text}")
        print(f"    top_label: {scores['top_label']}")
        for key, value in scores.items():
            if key == "top_label":
                continue
            print(f"    {key}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()
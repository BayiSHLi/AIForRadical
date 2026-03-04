import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


INPUT_PATH = Path("generated_samples/samples_79x4x20.jsonl")
OUT_DIR = Path("generated_samples")
SUMMARY_JSON = OUT_DIR / "samples_79x4x20_diversity_summary.json"
PAIR_CSV = OUT_DIR / "samples_79x4x20_pair_stats.csv"
INDICATOR_CSV = OUT_DIR / "samples_79x4x20_indicator_stats.csv"
RADICALITY_CSV = OUT_DIR / "samples_79x4x20_radicality_stats.csv"
TOP_DUPLICATES_CSV = OUT_DIR / "samples_79x4x20_top_duplicates.csv"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str):
    # keep simple and robust for mixed text
    return re.findall(r"\b\w+\b", text.lower())


def distinct_metrics(texts):
    all_tokens = []
    all_bigrams = []
    for t in texts:
        tokens = tokenize(t)
        all_tokens.extend(tokens)
        if len(tokens) >= 2:
            all_bigrams.extend(list(zip(tokens, tokens[1:])))

    unique_tokens = len(set(all_tokens))
    unique_bigrams = len(set(all_bigrams))
    total_tokens = len(all_tokens)
    total_bigrams = len(all_bigrams)

    ttr = (unique_tokens / total_tokens) if total_tokens else 0.0
    distinct_1 = (unique_tokens / total_tokens) if total_tokens else 0.0
    distinct_2 = (unique_bigrams / total_bigrams) if total_bigrams else 0.0

    return {
        "token_count": total_tokens,
        "unique_token_count": unique_tokens,
        "bigram_count": total_bigrams,
        "unique_bigram_count": unique_bigrams,
        "ttr": ttr,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
    }


def group_stats(rows):
    texts = [r["Content"] for r in rows]
    normalized = [normalize_text(t) for t in texts]

    n = len(texts)
    unique_exact = len(set(texts))
    unique_norm = len(set(normalized))

    char_lens = [len(t) for t in texts]
    word_lens = [len(tokenize(t)) for t in texts]

    d = distinct_metrics(texts)

    return {
        "count": n,
        "unique_exact": unique_exact,
        "unique_normalized": unique_norm,
        "exact_duplicate_rate": 1 - (unique_exact / n if n else 1),
        "normalized_duplicate_rate": 1 - (unique_norm / n if n else 1),
        "avg_char_len": mean(char_lens) if char_lens else 0,
        "median_char_len": median(char_lens) if char_lens else 0,
        "avg_word_len": mean(word_lens) if word_lens else 0,
        "median_word_len": median(word_lens) if word_lens else 0,
        **d,
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    rows = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)

    # group buckets
    by_indicator = defaultdict(list)
    by_radicality = defaultdict(list)
    by_pair = defaultdict(list)

    content_counter = Counter()
    normalized_counter = Counter()

    for r in rows:
        indicator = r.get("indicator", "")
        radicality = r.get("Radicality", "")
        content = r.get("Content", "")

        by_indicator[indicator].append(r)
        by_radicality[radicality].append(r)
        by_pair[(indicator, radicality)].append(r)

        content_counter[content] += 1
        normalized_counter[normalize_text(content)] += 1

    overall = group_stats(rows)

    # top duplicates
    top_exact_duplicates = [
        {"content": k, "count": v}
        for k, v in content_counter.most_common(50)
        if v > 1
    ]
    top_norm_duplicates = [
        {"normalized_content": k, "count": v}
        for k, v in normalized_counter.most_common(50)
        if v > 1
    ]

    # pair stats
    pair_records = []
    for (indicator, radicality), group in by_pair.items():
        s = group_stats(group)
        pair_records.append({
            "indicator": indicator,
            "Radicality": radicality,
            **s,
        })
    pair_records.sort(key=lambda x: (x["indicator"], x["Radicality"]))

    # indicator stats
    indicator_records = []
    for indicator, group in by_indicator.items():
        s = group_stats(group)
        indicator_records.append({
            "indicator": indicator,
            **s,
        })
    indicator_records.sort(key=lambda x: x["indicator"])

    # radicality stats
    radicality_order = ["Neutral", "Low", "Medium", "High"]
    radicality_records = []
    for radicality, group in by_radicality.items():
        s = group_stats(group)
        radicality_records.append({
            "Radicality": radicality,
            **s,
        })
    radicality_records.sort(key=lambda x: radicality_order.index(x["Radicality"]) if x["Radicality"] in radicality_order else 99)

    # export csv
    def write_csv(path, records):
        if not records:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    write_csv(PAIR_CSV, pair_records)
    write_csv(INDICATOR_CSV, indicator_records)
    write_csv(RADICALITY_CSV, radicality_records)

    with TOP_DUPLICATES_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "count", "text"])
        writer.writeheader()
        for item in top_exact_duplicates:
            writer.writerow({"type": "exact", "count": item["count"], "text": item["content"]})
        for item in top_norm_duplicates:
            writer.writerow({"type": "normalized", "count": item["count"], "text": item["normalized_content"]})

    summary = {
        "input_file": str(INPUT_PATH),
        "total_samples": total,
        "group_counts": {
            "indicators": len(by_indicator),
            "radicality_levels": len(by_radicality),
            "indicator_radicality_pairs": len(by_pair),
        },
        "overall": overall,
        "top_exact_duplicates_count": len(top_exact_duplicates),
        "top_normalized_duplicates_count": len(top_norm_duplicates),
        "output_files": {
            "pair_stats": str(PAIR_CSV),
            "indicator_stats": str(INDICATOR_CSV),
            "radicality_stats": str(RADICALITY_CSV),
            "top_duplicates": str(TOP_DUPLICATES_CSV),
        },
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

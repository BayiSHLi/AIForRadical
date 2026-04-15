"""
Build an external dataset pipeline for Simulation retrieval.

Pipeline steps:
1) Load external dataset rows (MIWS).
2) Assign indicators using indicator definitions.
3) Preserve dataset radical annotations when available.
4) Export processed rows to JSONL.
5) Build an independent retrieval index for external data.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jsonlines
import pandas as pd

from full_indicators import FULL_INDICATORS
from simulator_config import INDICATORS


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_OUTPUT_JSONL = str(DATA_DIR / "external_data" / "miws_processed.jsonl")
DEFAULT_INDEX_DIR = str(DATA_DIR / "external_data_index")


@dataclass
class IndicatorProfile:
    indicator_id: str
    factor: str
    description: str
    keywords: List[str]


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


def _build_indicator_profiles() -> List[IndicatorProfile]:
    profiles: List[IndicatorProfile] = []
    for ind_id, base in FULL_INDICATORS.items():
        ext = INDICATORS.get(ind_id, {})
        sample_keywords = ext.get("sample_keywords", []) or []

        keyword_set = []
        for kw in sample_keywords:
            term = _normalize_text(kw)
            if term and term not in keyword_set:
                keyword_set.append(term)

        desc_tokens = _tokenize(base.get("description", ""))
        # Keep only meaningful description tokens as soft anchors.
        for token in desc_tokens:
            if len(token) >= 5 and token not in keyword_set:
                keyword_set.append(token)
            if len(keyword_set) >= 16:
                break

        profiles.append(
            IndicatorProfile(
                indicator_id=ind_id,
                factor=str(base.get("factor", "")),
                description=str(base.get("description", "")),
                keywords=keyword_set,
            )
        )
    return profiles


def _score_indicator(text: str, profile: IndicatorProfile) -> float:
    tokens = set(_tokenize(text))
    if not tokens:
        return 0.0

    overlap = 0.0
    for kw in profile.keywords:
        if not kw:
            continue
        if " " in kw:
            if kw in text:
                overlap += 1.5
        elif kw in tokens:
            overlap += 1.0

    # Encourage semantic grounding via factor and description token overlap.
    factor_tokens = set(_tokenize(profile.factor))
    desc_tokens = set(_tokenize(profile.description))
    factor_hits = len(tokens.intersection(factor_tokens))
    desc_hits = len(tokens.intersection(desc_tokens))

    score = overlap + factor_hits * 0.35 + min(desc_hits, 6) * 0.20
    return float(score)


def assign_indicators(text: str, profiles: List[IndicatorProfile], top_k: int) -> List[Tuple[str, float]]:
    normalized = _normalize_text(text)
    scored: List[Tuple[str, float]] = []
    for profile in profiles:
        score = _score_indicator(normalized, profile)
        if score > 0:
            scored.append((profile.indicator_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _load_miws(miws_dir: Path) -> pd.DataFrame:
    isis_path = miws_dir / "ISIS_labels.csv"
    ws_path = miws_dir / "WS_Labels.csv"

    if not isis_path.exists() and not ws_path.exists():
        raise FileNotFoundError(f"No MIWS CSV found in {miws_dir}")

    frames: List[pd.DataFrame] = []

    if isis_path.exists():
        df_isis = pd.read_csv(isis_path)
        df_isis["dataset_partition"] = "MIWS_ISIS"
        df_isis["source_text"] = df_isis.get("tweets", "")
        frames.append(df_isis)

    if ws_path.exists():
        df_ws = pd.read_csv(ws_path)
        df_ws["dataset_partition"] = "MIWS_WS"
        df_ws["source_text"] = df_ws.get("Text", "")
        frames.append(df_ws)

    merged = pd.concat(frames, ignore_index=True)
    return merged


def _safe_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_processed_records(
    miws_df: pd.DataFrame,
    indicator_top_k: int,
    max_rows: int,
) -> List[Dict]:
    profiles = _build_indicator_profiles()

    if max_rows > 0:
        miws_df = miws_df.head(max_rows)

    records: List[Dict] = []
    now = datetime.utcnow().isoformat()

    for idx, row in miws_df.iterrows():
        raw_text = _safe_value(row.get("source_text", ""))
        cleaned_text = _safe_value(row.get("cleaned_text", ""))
        content_for_assignment = cleaned_text or raw_text
        if not content_for_assignment:
            continue

        assigned = assign_indicators(content_for_assignment, profiles, indicator_top_k)
        assigned_ids = [item[0] for item in assigned]

        record = {
            "external_id": f"miws_{idx}",
            "source_dataset": _safe_value(row.get("dataset_partition", "MIWS")),
            "source": _safe_value(row.get("Source", "")),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "annotation_label": _safe_value(row.get("Labels", "")),
            "annotation_topic": _safe_value(row.get("Topics", "")),
            "assigned_indicators": assigned_ids,
            "assigned_indicator_scores": [
                {"indicator": ind_id, "score": round(score, 4)} for ind_id, score in assigned
            ],
            "processed_at": now,
        }
        records.append(record)

    return records


def save_processed_jsonl(records: List[Dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_jsonl, mode="w") as writer:
        for record in records:
            writer.write(record)


def _build_documents_for_index(records: List[Dict]):
    from llama_index.core import Document

    docs = []
    for rec in records:
        text = (
            f"DATASET: {rec['source_dataset']}\n"
            f"SOURCE: {rec['source'] or 'N/A'}\n"
            f"ANNOTATION_LABEL: {rec['annotation_label'] or 'N/A'}\n"
            f"ANNOTATION_TOPIC: {rec['annotation_topic'] or 'N/A'}\n"
            f"ASSIGNED_INDICATORS: {', '.join(rec['assigned_indicators']) if rec['assigned_indicators'] else 'N/A'}\n"
            f"CONTENT: {rec['raw_text'] or rec['cleaned_text']}"
        )

        metadata = {
            "external_id": rec["external_id"],
            "source_dataset": rec["source_dataset"],
            "source": rec["source"],
            "annotation_label": rec["annotation_label"],
            "annotation_topic": rec["annotation_topic"],
            "assigned_indicators": rec["assigned_indicators"],
        }

        docs.append(Document(text=text, metadata=metadata, doc_id=rec["external_id"]))

    return docs


def build_external_index(records: List[Dict], index_dir: Path) -> Dict:
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "index_dir": str(index_dir),
        "record_count": len(records),
        "created_at": datetime.utcnow().isoformat(),
        "status": "unknown",
        "message": "",
    }

    if not records:
        manifest["status"] = "empty"
        manifest["message"] = "No processed records available to index."
        (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    try:
        from llama_index.core import Settings, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as exc:
        manifest["status"] = "dependency_missing"
        manifest["message"] = (
            "Index build skipped because llama-index dependencies are unavailable: "
            f"{exc}"
        )
        (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.embed_model = embed_model

    docs = _build_documents_for_index(records)
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes=nodes, storage_context=storage, embed_model=embed_model, show_progress=True)
    index.storage_context.persist(str(index_dir))

    (index_dir / "metadata.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["status"] = "ok"
    manifest["message"] = "External index built successfully."
    manifest["node_count"] = len(nodes)
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external dataset retrieval pipeline for Simulation")
    parser.add_argument("--miws-dir", required=True, help="Path to MIWS_Dataset_Standard folder")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Processed external records JSONL path")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="External retrieval index directory")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for smoke test; 0 means all")
    parser.add_argument("--indicator-top-k", type=int, default=3, help="Number of indicators to assign per record")
    return parser.parse_args()


def _resolve_data_output_path(path_arg: str) -> Path:
    """Resolve output/index path to project data folder when a relative path is provided."""
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return PROJECT_ROOT / path
    return DATA_DIR / path


def main() -> int:
    args = parse_args()

    miws_dir = Path(args.miws_dir)
    output_jsonl = _resolve_data_output_path(args.output_jsonl)
    index_dir = _resolve_data_output_path(args.index_dir)

    df = _load_miws(miws_dir)
    records = build_processed_records(
        miws_df=df,
        indicator_top_k=max(1, args.indicator_top_k),
        max_rows=max(0, args.max_rows),
    )

    save_processed_jsonl(records, output_jsonl)
    manifest = build_external_index(records, index_dir)

    print("=" * 80)
    print("EXTERNAL DATA PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Input rows loaded: {len(df)}")
    print(f"Processed rows: {len(records)}")
    print(f"Processed JSONL: {output_jsonl}")
    print(f"Index directory: {index_dir}")
    print(f"Index status: {manifest.get('status')}")
    print(f"Index message: {manifest.get('message')}")
    print("=" * 80)

    if manifest.get("status") in {"ok", "empty", "dependency_missing"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

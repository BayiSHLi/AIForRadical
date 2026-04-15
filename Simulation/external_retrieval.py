"""
External dataset retrieval interface for Simulation.

This module loads the external index built by external_dataset_pipeline.py and
returns retrieval snippets for prompt grounding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


class ExternalDataRetriever:
    """Lightweight retrieval wrapper over an external llama-index store."""

    def __init__(self, index_dir: Optional[str] = None, similarity_top_k: int = 3):
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        default_index_dir = project_root / "data" / "external_data_index"
        self.index_dir = Path(index_dir) if index_dir else default_index_dir
        self.similarity_top_k = similarity_top_k
        self.is_available = False
        self._retriever = None

        self._try_load_index()

    def _try_load_index(self) -> None:
        required_files = [
            self.index_dir / "docstore.json",
            self.index_dir / "index_store.json",
            self.index_dir / "default__vector_store.json",
        ]
        if not self.index_dir.exists() or not all(p.exists() for p in required_files):
            self.is_available = False
            return

        try:
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except Exception:
            self.is_available = False
            return

        try:
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
            Settings.embed_model = embed_model
            storage = StorageContext.from_defaults(persist_dir=str(self.index_dir))
            index = load_index_from_storage(storage, embed_model=embed_model)
            self._retriever = index.as_retriever(similarity_top_k=self.similarity_top_k)
            self.is_available = True
        except Exception:
            self.is_available = False
            self._retriever = None

    def build_query(self, indicator: str, radicality: str, indicator_description: str = "") -> str:
        desc = indicator_description.strip() if indicator_description else ""
        return (
            f"Find external social-media examples related to indicator '{indicator}' "
            f"at radicality level '{radicality}'. "
            f"Indicator meaning: {desc}"
        )

    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        if not self.is_available or self._retriever is None:
            return []

        k = top_k if top_k is not None else self.similarity_top_k

        try:
            source_nodes = self._retriever.retrieve(query_text)
        except Exception:
            return []

        rows: List[Dict] = []
        for node in source_nodes[:k]:
            metadata = getattr(node, "metadata", {}) or {}
            text = getattr(node, "text", "") or ""
            score = getattr(node, "score", None)
            rows.append(
                {
                    "text": text,
                    "score": float(score) if score is not None else None,
                    "metadata": metadata,
                }
            )
        return rows

    def format_for_prompt(self, results: List[Dict], max_items: int = 3) -> str:
        if not results:
            return ""

        lines: List[str] = []
        for i, row in enumerate(results[:max_items], start=1):
            metadata = row.get("metadata", {}) or {}
            label = metadata.get("annotation_label", "")
            topic = metadata.get("annotation_topic", "")
            dataset = metadata.get("source_dataset", "")
            indicators = metadata.get("assigned_indicators", [])
            indicator_text = ", ".join(indicators[:3]) if indicators else "N/A"

            snippet = " ".join((row.get("text") or "").split())
            snippet = snippet[:220]

            lines.append(
                f"- Ref#{i} | dataset={dataset or 'N/A'} | label={label or 'N/A'} | "
                f"topic={topic or 'N/A'} | indicators={indicator_text}\n"
                f"  snippet: {snippet}"
            )

        return "\n".join(lines)

from __future__ import annotations

import re
from typing import Any


def split_numbered_passages(text: str) -> list[str]:
    """Split `[1] ... [2] ...` evidence while retaining each source label verbatim."""
    parts = re.split(r"(?=\[\d+\])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_corpus(dataset: Any, parse_numbered_docs: bool = False) -> list[dict[str, Any]]:
    """Build retrieval units from answer-oriented rows.

    A row is not assumed to be an independent web page: by default its full `doc`
    evidence bundle is one unit. Optional parsing turns explicitly numbered passages
    into units, preserving the labels so evidence provenance remains visible.
    """
    corpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_id, row in enumerate(dataset):
        document = str(row.get("doc") or "").strip()
        query = str(row.get("query") or "").strip()
        if not document or not query:
            continue
        passages = split_numbered_passages(document) if parse_numbered_docs else [document]
        for passage_id, text in enumerate(passages):
            key = text.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            corpus.append({"doc_id": f"r{row_id}_p{passage_id}", "row_id": row_id,
                           "text": text, "source_query": query})
    if not corpus:
        raise ValueError("No non-empty, unique documents were available for retrieval.")
    return corpus

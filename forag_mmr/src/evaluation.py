from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .prompts import JUDGE_PROMPT, render_evidence


def evaluate_redundancy(documents: list[dict], embedding_lookup: dict[str, np.ndarray]) -> dict[str, float]:
    vectors = np.array([embedding_lookup[d["doc_id"]] for d in documents])
    if len(vectors) < 2:
        return {"average_pairwise_similarity": 0.0, "maximum_pairwise_similarity": 0.0,
                "retrieval_redundancy": 0.0, "retrieval_diversity": 1.0}
    pairwise = vectors @ vectors.T
    values = pairwise[np.triu_indices(len(vectors), k=1)]
    average = float(values.mean())
    return {"average_pairwise_similarity": average, "maximum_pairwise_similarity": float(values.max()),
            "retrieval_redundancy": average, "retrieval_diversity": float(1 - average)}


def evaluate_retrieval(documents: list[dict], embedding_lookup: dict[str, np.ndarray]) -> dict[str, float]:
    metrics = evaluate_redundancy(documents, embedding_lookup)
    metrics["average_query_similarity"] = float(np.mean([d["query_similarity"] for d in documents])) if documents else 0.0
    return metrics


def _parse_judge(raw: str) -> dict:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        result = json.loads(raw[start:end])
        return {key: max(1, min(5, int(result[key]))) for key in ("coherence", "helpfulness", "factuality")} | {"reason": str(result.get("reason", ""))}
    except Exception as exc:
        return {"coherence": None, "helpfulness": None, "factuality": None, "reason": f"Malformed judge JSON: {exc}"}


def evaluate_answer(client, question: str, documents: list[dict], answer: str, reference_answer: str) -> dict:
    raw = client.complete(JUDGE_PROMPT.format(question=question, evidence=render_evidence(documents),
                                               reference_answer=reference_answer, answer=answer))
    return _parse_judge(raw)


def summarize(retrieval_df: pd.DataFrame, answer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [("Coherence", "coherence"), ("Helpfulness", "helpfulness"), ("Factuality", "factuality"),
               ("Retrieval Redundancy", "retrieval_redundancy"), ("Average Query Similarity", "average_query_similarity"),
               ("Average Answer Length", "answer_length")]
    for label, column in metrics:
        base = pd.to_numeric((answer_df if column in answer_df else retrieval_df).query("system == 'baseline'")[column], errors="coerce").mean()
        mmr = pd.to_numeric((answer_df if column in answer_df else retrieval_df).query("system == 'mmr'")[column], errors="coerce").mean()
        rows.append({"Metric": label, "Baseline": base, "MMR": mmr, "Improvement": mmr - base})
    return pd.DataFrame(rows)

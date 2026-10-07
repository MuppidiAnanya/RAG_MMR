from __future__ import annotations

import numpy as np


def maximal_marginal_relevance(candidate_embeddings: np.ndarray, query_scores: np.ndarray,
                                final_k: int, lambda_mult: float = 0.7) -> list[dict]:
    """Select candidate indices with MMR using normalized-vector cosine similarity."""
    if not 0 <= lambda_mult <= 1:
        raise ValueError("MMR lambda must be between 0 and 1.")
    n = len(candidate_embeddings)
    if n == 0:
        return []
    chosen: list[int] = [int(np.argmax(query_scores))]
    metadata = [{"candidate_index": chosen[0], "redundancy_similarity": 0.0,
                 "mmr_score": float(query_scores[chosen[0]])}]
    while len(chosen) < min(final_k, n):
        remaining = [i for i in range(n) if i not in chosen]
        redundancy = candidate_embeddings[remaining] @ candidate_embeddings[chosen].T
        max_redundancy = redundancy.max(axis=1)
        scores = lambda_mult * query_scores[remaining] - (1 - lambda_mult) * max_redundancy
        pos = int(np.argmax(scores))
        selected = remaining[pos]
        chosen.append(selected)
        metadata.append({"candidate_index": selected, "redundancy_similarity": float(max_redundancy[pos]),
                         "mmr_score": float(scores[pos])})
    return metadata

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from .mmr import maximal_marginal_relevance


class FaissRetriever:
    def __init__(self, corpus: list[dict[str, Any]], embeddings: np.ndarray, embedder: Any):
        if len(corpus) != len(embeddings):
            raise ValueError("Corpus and embedding counts do not match.")
        self.corpus, self.embeddings, self.embedder = corpus, embeddings.astype("float32"), embedder
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _candidates(self, query: str, top_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        query_embedding = self.embedder.encode([query])[0]
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(top_n, len(self.corpus)))
        valid = indices[0] >= 0
        return indices[0][valid], scores[0][valid], query_embedding

    def retrieve_baseline(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ids, scores, _ = self._candidates(query, top_k)
        return [{**self.corpus[int(doc_idx)], "query_similarity": float(score), "rank": rank + 1}
                for rank, (doc_idx, score) in enumerate(zip(ids, scores))]

    def retrieve_mmr(self, query: str, candidate_k: int = 10, final_k: int = 5,
                     lambda_mult: float = 0.7) -> list[dict[str, Any]]:
        ids, scores, _ = self._candidates(query, candidate_k)
        details = maximal_marginal_relevance(self.embeddings[ids], scores, final_k, lambda_mult)
        selected = []
        for rank, detail in enumerate(details, 1):
            local = detail["candidate_index"]
            selected.append({**self.corpus[int(ids[local])], "query_similarity": float(scores[local]),
                             "redundancy_similarity": detail["redundancy_similarity"],
                             "mmr_score": detail["mmr_score"], "rank": rank})
        return selected

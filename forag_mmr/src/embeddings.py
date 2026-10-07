from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                 normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def embedding_cache_valid(path: Path, expected_count: int) -> bool:
    return path.exists() and np.load(path, mmap_mode="r").shape[0] == expected_count

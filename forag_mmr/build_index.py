from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data_loader import inspect_dataset, load_webglm_oe, print_inspection
from src.embeddings import Embedder, embedding_cache_valid
from src.preprocessing import build_corpus
from src.utils import Config, project_root, set_seed, write_json


def main() -> None:
    cfg = Config(); set_seed(cfg.seed)
    root = project_root(); artifacts = root / "artifacts"
    ds = load_webglm_oe(cfg.dataset_name, cfg.dataset_size, cfg.seed)
    report = inspect_dataset(ds); print_inspection(report)
    corpus = build_corpus(ds, cfg.parse_numbered_docs)
    embeddings_path = artifacts / "document_embeddings.npy"
    if embedding_cache_valid(embeddings_path, len(corpus)):
        print(f"Using cached embeddings at {embeddings_path}")
    else:
        print(f"Embedding {len(corpus)} retrieval units…")
        vectors = Embedder(cfg.embedding_model).encode([item["text"] for item in corpus])
        artifacts.mkdir(parents=True, exist_ok=True); np.save(embeddings_path, vectors)
    write_json(artifacts / "metadata.json", {"config": cfg.to_dict(), "dataset_inspection": report, "corpus": corpus})
    print(f"Saved {len(corpus)} corpus items and embeddings under {artifacts}")


if __name__ == "__main__":
    main()

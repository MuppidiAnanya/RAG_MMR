from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Config:
    dataset_name: str = os.getenv("DATASET_NAME", "forag/webglm_oe")
    dataset_size: int | None = int(os.getenv("DATASET_SIZE", "5000")) if os.getenv("DATASET_SIZE", "5000").lower() != "none" else None
    eval_size: int = int(os.getenv("EVAL_SIZE", "100"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retrieval_top_n: int = int(os.getenv("RETRIEVAL_TOP_N", "10"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
    parse_numbered_docs: bool = os.getenv("PARSE_NUMBERED_DOCS", "false").lower() == "true"
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1200"))
    seed: int = int(os.getenv("SEED", "42"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

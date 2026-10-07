from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

REQUIRED_FIELDS = {"query", "doc", "structure", "outline", "answer"}


def load_webglm_oe(dataset_name: str, dataset_size: int | None, seed: int) -> Dataset:
    ds = load_dataset(dataset_name, split="train")
    missing = REQUIRED_FIELDS - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    # Shuffle before selecting so a small prototype is a reproducible sample, not a prefix.
    return ds.shuffle(seed=seed).select(range(min(dataset_size, len(ds)))) if dataset_size else ds


def inspect_dataset(ds: Dataset, sample_count: int = 2) -> dict[str, Any]:
    frame = ds.to_pandas()
    report = {
        "rows": len(ds),
        "columns": ds.column_names,
        "missing_values": frame.isna().sum().to_dict(),
        "average_document_length": float(frame["doc"].fillna("").astype(str).str.len().mean()),
        "average_query_length": float(frame["query"].fillna("").astype(str).str.len().mean()),
        "samples": [{key: str(value)[:1000] for key, value in ds[i].items()} for i in range(min(sample_count, len(ds)))],
    }
    return report


def print_inspection(report: dict[str, Any]) -> None:
    print(f"Rows: {report['rows']}\nColumns: {report['columns']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Average document length: {report['average_document_length']:.1f}")
    print(f"Average query length: {report['average_query_length']:.1f}")
    print("Sample records:")
    for sample in report["samples"]:
        print(sample)

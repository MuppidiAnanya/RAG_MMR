from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load_webglm_oe
from src.embeddings import Embedder
from src.evaluation import evaluate_answer, evaluate_retrieval, summarize
from src.faiss_retriever import FaissRetriever
from src.generator import generate_outline_and_answer
from src.llm_client import LLMClient
from src.utils import Config, project_root, read_json, set_seed, write_json


def plot_summary(summary: pd.DataFrame, plots: Path) -> None:
    import matplotlib.pyplot as plt
    plots.mkdir(parents=True, exist_ok=True)
    requested = ["Retrieval Redundancy", "Average Query Similarity", "Coherence", "Helpfulness", "Factuality"]
    for metric in requested:
        row = summary[summary["Metric"] == metric].iloc[0]
        fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(["Baseline", "MMR"], [row["Baseline"], row["MMR"]])
        ax.set_title(metric); ax.set_ylabel("Score"); fig.tight_layout(); fig.savefig(plots / f"{metric.lower().replace(' ', '_')}.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--eval-size", type=int); args = parser.parse_args()
    cfg = Config(); cfg.eval_size = args.eval_size or cfg.eval_size; set_seed(cfg.seed)
    root = project_root(); metadata = read_json(root / "artifacts" / "metadata.json")
    if not metadata:
        raise FileNotFoundError("Build the index first: python build_index.py")
    corpus = metadata["corpus"]; embeddings = np.load(root / "artifacts" / "document_embeddings.npy")
    retriever = FaissRetriever(corpus, embeddings, Embedder(cfg.embedding_model))
    lookup = {doc["doc_id"]: embeddings[i] for i, doc in enumerate(corpus)}
    dataset = load_webglm_oe(cfg.dataset_name, cfg.dataset_size, cfg.seed)
    eval_rows = [dataset[i] for i in range(min(cfg.eval_size, len(dataset)))]
    client = None if args.retrieval_only else LLMClient(cfg.llm_provider, cfg.llm_model, cfg.temperature, cfg.max_tokens)
    retrieval_records, answer_records, examples = [], [], []
    cache_path = root / "results" / "generation_cache.json"
    generation_cache = read_json(cache_path, {})
    for row_id, row in enumerate(tqdm(eval_rows, desc="Evaluating")):
        question = str(row["query"])
        systems = {"baseline": retriever.retrieve_baseline(question, cfg.final_top_k),
                   "mmr": retriever.retrieve_mmr(question, cfg.retrieval_top_n, cfg.final_top_k, cfg.mmr_lambda)}
        example = {"question": question}
        for name, docs in systems.items():
            metrics = evaluate_retrieval(docs, lookup)
            retrieval_records.append({"row_id": row_id, "system": name, **metrics, "documents": docs})
            example[f"{name}_documents"] = docs; example[f"{name}_redundancy"] = metrics["retrieval_redundancy"]
            if client:
                key_source = f"{cfg.llm_model}|{name}|{question}|" + "|".join(d["doc_id"] for d in docs)
                cache_key = hashlib.sha256(key_source.encode()).hexdigest()
                cached = generation_cache.get(cache_key)
                if cached:
                    outline, answer, scores = cached["outline"], cached["answer"], cached["scores"]
                else:
                    outline, answer = generate_outline_and_answer(client, question, docs)
                    scores = evaluate_answer(client, question, docs, answer, str(row.get("answer", "")))
                    generation_cache[cache_key] = {"outline": outline, "answer": answer, "scores": scores}
                answer_records.append({"row_id": row_id, "system": name, "answer_length": len(answer), "outline": outline, "answer": answer, **scores})
                example[f"{name}_outline"] = outline; example[f"{name}_answer"] = answer
        if row_id < 10: examples.append(example)
    results = root / "results"; results.mkdir(exist_ok=True)
    retrieval_df = pd.DataFrame(retrieval_records); retrieval_df.to_csv(results / "retrieval_results.csv", index=False)
    write_json(results / "examples.json", examples)
    if client:
        write_json(cache_path, generation_cache)
    if answer_records:
        answer_df = pd.DataFrame(answer_records); answer_df.to_csv(results / "answer_results.csv", index=False)
        summary = summarize(retrieval_df, answer_df); summary.to_csv(results / "summary.csv", index=False); plot_summary(summary, results / "plots")
    else:
        print("Retrieval-only run completed. Answer scores, summary, and answer plots require an LLM API key.")


if __name__ == "__main__":
    main()

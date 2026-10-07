from __future__ import annotations

import numpy as np
import streamlit as st

from src.embeddings import Embedder
from src.evaluation import evaluate_retrieval
from src.faiss_retriever import FaissRetriever
from src.generator import generate_outline_and_answer
from src.llm_client import LLMClient
from src.utils import Config, project_root, read_json

st.set_page_config(page_title="MMR-FoRAG", layout="wide")
st.title("FoRAG-inspired long-form QA with diversity-aware MMR retrieval")
st.caption("Compare similarity-only retrieval against MMR. Scores are experiment outputs, not claims of improvement.")

@st.cache_resource
def load_retriever():
    cfg = Config(); root = project_root(); metadata = read_json(root / "artifacts" / "metadata.json")
    if not metadata: return None
    corpus = metadata["corpus"]; vectors = np.load(root / "artifacts" / "document_embeddings.npy")
    return FaissRetriever(corpus, vectors, Embedder(cfg.embedding_model)), {d["doc_id"]: vectors[i] for i, d in enumerate(corpus)}

loaded = load_retriever()
if not loaded:
    st.warning("No index found. Run `python build_index.py` from the project directory first."); st.stop()
retriever, lookup = loaded; cfg = Config()
question = st.text_area("QUESTION", placeholder="Ask a long-form question…")
if st.button("Compare retrieval", type="primary") and question.strip():
    systems = {"Baseline retrieval": retriever.retrieve_baseline(question, cfg.final_top_k),
               "MMR retrieval": retriever.retrieve_mmr(question, cfg.retrieval_top_n, cfg.final_top_k, cfg.mmr_lambda)}
    cols = st.columns(2)
    for column, (label, docs) in zip(cols, systems.items()):
        with column:
            st.subheader(label)
            m = evaluate_retrieval(docs, lookup)
            st.metric("Redundancy", f"{m['retrieval_redundancy']:.3f}")
            for doc in docs:
                title = f"#{doc['rank']} · similarity {doc['query_similarity']:.3f}"
                if "mmr_score" in doc: title += f" · MMR {doc['mmr_score']:.3f}"
                with st.expander(title): st.write(doc["text"])
    if st.checkbox("Generate outlines and answers (requires OPENAI_API_KEY)"):
        client = LLMClient(cfg.llm_provider, cfg.llm_model, cfg.temperature, cfg.max_tokens)
        for label, docs in systems.items():
            try:
                outline, answer = generate_outline_and_answer(client, question, docs)
                st.subheader(f"{label}: outline"); st.write(outline)
                st.subheader(f"{label}: answer"); st.write(answer)
            except Exception as exc: st.error(f"Generation failed: {exc}")

summary = read_json(project_root() / "results" / "summary.json")
if not summary:
    try:
        import pandas as pd
        summary_path = project_root() / "results" / "summary.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else None
    except Exception:
        summary = None
st.subheader("COMPARISON")
if summary is not None:
    st.caption("Aggregate held-out experiment scores; they are not generated for the ad-hoc question above.")
    st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.info("Run a full experiment to display aggregate coherence, helpfulness, factuality, and redundancy here.")

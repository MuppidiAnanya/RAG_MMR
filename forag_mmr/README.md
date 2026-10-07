# FoRAG-inspired long-form QA with diversity-aware MMR retrieval

This research prototype tests one contribution: **diversity-aware retrieval using Sentence-BERT, FAISS, and Maximal Marginal Relevance (MMR)**. It is inspired by FoRAG's evidence-to-outline-to-answer framing; it is not a reproduction of FoRAG RLHF, reward-model training, or fine-tuning.

## Problem and research questions

Similarity-only retrieval can return overlapping evidence. This project compares it with MMR reranking on the same WebGLM-OE questions, evidence count, prompts, model, and generation settings.

- RQ1: Does MMR reduce evidence redundancy?
- RQ2: Does it preserve sufficient query relevance?
- RQ3: Does it improve coherence, helpfulness, or factuality?
- RQ4: Does evidence diversity lead to better long-form answers?

H1 expects lower redundancy; H2 expects competitive relevance. H3 is a hypothesis, not an assumed result. Never compare these outputs directly to published FoRAG metrics as a fair benchmark.

## Architecture

```text
Question → MiniLM embedding → FAISS cosine candidates ─→ top-K baseline evidence
                                                └→ MMR (λ=0.7) → top-K diverse evidence
Each evidence set → identical outline prompt → identical answer prompt → optional blind-style LLM judge
```

Documents come from WebGLM-OE's answer-oriented rows. A row's `doc` is an evidence bundle, rather than automatically a standalone web document. The default preserves the whole bundle as one unit. Set `PARSE_NUMBERED_DOCS=true` to split visibly numbered passages while preserving source labels; this is a different corpus construction and should be reported.

## Dataset and configuration

The loader requires actual `query`, `doc`, `structure`, `outline`, and `answer` fields and prints rows, fields, samples, missing values, and average lengths. Reference `outline` and `answer` are never passed into generation. The answer is used only by the evaluator.

Copy `.env.example` into your environment. Defaults are `DATASET_SIZE=5000`, `EVAL_SIZE=100`, seed 42, top-N 10, top-K 5, λ 0.7. Set `DATASET_SIZE=None` to use the full corpus. Start with `DATASET_SIZE=500` and `EVAL_SIZE=10`.

## Installation

```bash
cd forag_mmr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For outline generation, answering, and LLM judging, configure one provider and its matching environment variable. `LLM_PROVIDER=openai` uses `OPENAI_API_KEY`; `LLM_PROVIDER=groq` uses `GROQ_API_KEY` through Groq's OpenAI-compatible endpoint. The default Groq model is `llama-3.3-70b-versatile`; replace it if your Groq account exposes a different model. The retrieval-only pipeline does not need an API key.

## Run

```bash
# Downloads/inspects data, constructs corpus, caches normalized embeddings and metadata
python build_index.py

# Fair retrieval comparison only (writes retrieval_results.csv and examples.json)
python run_experiment.py --retrieval-only --eval-size 10

# Full comparison (requires OPENAI_API_KEY; creates answer_results, summary, and plots)
python run_experiment.py --eval-size 100

# Interactive UI
streamlit run app.py
```

`artifacts/document_embeddings.npy` prevents re-encoding the corpus and `artifacts/metadata.json` records its configuration/corpus. Rebuild after changing dataset size, parsing mode, or embedding model.

## Metrics and outputs

`results/retrieval_results.csv` includes query similarity, average/maximum pairwise cosine similarity, redundancy (the primary metric: lower is better), and diversity (1 − redundancy). Full runs add `results/answer_results.csv`, `results/summary.csv`, five metric plots under `results/plots/`, and ten qualitative examples in `results/examples.json`.

The judge returns structured 1–5 coherence, helpfulness, and evidence-supported factuality scores. Its prompt does not identify baseline versus MMR. API errors and malformed judge JSON are recorded rather than converted into invented scores.

## Limitations and future work

This uses a small embedding model and optional LLM-as-judge, both of which can bias conclusions. The corpus is derived from QA records and not a canonical web-page index. Future work could add human assessment, confidence intervals, multiple seeds/models, passage provenance, and a separately validated factuality evaluator.

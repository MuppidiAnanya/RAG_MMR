OUTLINE_PROMPT = """You are writing an evidence-grounded outline. Use only the supplied evidence.
QUESTION: {question}
EVIDENCE:\n{evidence}
Return a concise structured outline of the main points needed to answer the question. Do not add unsupported claims."""

ANSWER_PROMPT = """Write a coherent long-form answer to the question using only the supplied evidence and following the outline. Answer directly. Avoid unsupported claims, invented citations, and any mention of retrieval.
QUESTION: {question}
EVIDENCE:\n{evidence}
OUTLINE:\n{outline}
ANSWER:"""

JUDGE_PROMPT = """Evaluate one answer without inferring how it was produced. Score each criterion from 1 to 5. Factuality means support by the provided evidence. The reference answer is evaluation context, not evidence. Return ONLY valid JSON with coherence, helpfulness, factuality, and reason.
QUESTION: {question}
EVIDENCE:\n{evidence}
REFERENCE ANSWER:\n{reference_answer}
ANSWER TO EVALUATE:\n{answer}"""


def render_evidence(documents: list[dict]) -> str:
    return "\n\n".join(f"Document {i + 1}:\n{doc['text']}" for i, doc in enumerate(documents))

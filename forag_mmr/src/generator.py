from __future__ import annotations

from .prompts import ANSWER_PROMPT, OUTLINE_PROMPT, render_evidence


def generate_outline_and_answer(client, question: str, documents: list[dict]) -> tuple[str, str]:
    evidence = render_evidence(documents)
    outline = client.complete(OUTLINE_PROMPT.format(question=question, evidence=evidence))
    answer = client.complete(ANSWER_PROMPT.format(question=question, evidence=evidence, outline=outline))
    return outline, answer

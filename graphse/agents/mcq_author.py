"""MCQ author with deterministic validation.

The model proposes a question; code validates option count, distinct options,
a single valid key and a support reference into the supplied evidence. The
server then attaches the exact source sentence, so the model cannot alter the
quotation. Rejected candidates are regenerated a bounded number of times.
"""
from __future__ import annotations

import json

from ..retrieval.evidence import Citation, numbered_excerpts
from .llm import LLMClient

SYSTEM = (
    "Write one multiple-choice question that tests the named concept using ONLY the numbered excerpts. "
    'Return JSON: {"question": str, "options": [3-4 str], "correct_index": int, '
    '"explanation": str, "support_ref": int}.'
)


class GenerationRejected(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def validate_mcq(item: object, citations: list[Citation]) -> dict:
    if not isinstance(item, dict):
        raise ValueError("MCQ must be an object")
    for key in ("question", "explanation"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"missing {key}")
    options = item.get("options")
    if not isinstance(options, list) or len(options) not in (3, 4) or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError("needs 3-4 non-empty options")
    if len({o.strip().casefold() for o in options}) != len(options):
        raise ValueError("duplicate options")
    key = item.get("correct_index")
    if type(key) is not int or not 0 <= key < len(options):
        raise ValueError("invalid answer key")
    ref = item.get("support_ref")
    if type(ref) is not int or not 0 <= ref < len(citations):
        raise ValueError("support_ref outside the supplied evidence")
    return {"question": item["question"].strip(), "options": [o.strip() for o in options], "correct_index": key,
            "explanation": item["explanation"].strip(), "support_ref": ref,
            "support_quote": citations[ref].text}      # attached by the server, never by the model


class MCQAuthor:
    def __init__(self, llm: LLMClient, max_attempts: int = 3, temperature: float = 0.2):
        self.llm, self.max_attempts, self.temperature = llm, max_attempts, temperature

    def write(self, concept_name: str, citations: list[Citation]) -> dict:
        if not citations:
            raise GenerationRejected(["no source evidence for this concept"])
        prompt = f"Concept: {concept_name}\n\nExcerpts:\n{numbered_excerpts(citations)}"
        reasons = []
        for _ in range(self.max_attempts):
            raw = self.llm.complete(SYSTEM, [{"role": "user", "content": prompt}], self.temperature)
            try:
                return validate_mcq(json.loads(raw), citations)
            except (ValueError, json.JSONDecodeError) as exc:
                reasons.append(str(exc))
        raise GenerationRejected(reasons)

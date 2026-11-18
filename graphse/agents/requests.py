"""Rule-based request classification (no model call).

A learner message is resolved into an action, a target and an output format
before any agent is called. The reference system also handles a second
language and more actions; this abstraction keeps the core set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ACTIONS = ("meta", "repair", "walkthrough", "create_example", "simplify", "compare", "explain")


@dataclass(frozen=True)
class Request:
    action: str
    target: str
    output_format: str           # "prose", "code", "diagram" or "table"
    another: bool = False


def classify_request(question: str, has_current_asset: bool = False) -> Request:
    q = question.casefold().strip()
    fmt = ("diagram" if re.search(r"\b(diagram|flowchart)\b", q) else
           "table" if re.search(r"\btable\b", q) else
           "code" if re.search(r"\b(code|program)\b", q) else "prose")
    another = bool(re.search(r"\b(another|different|other)\b", q))
    if re.search(r"\b(who|what) are you\b|\bwhat can you do\b", q):
        return Request("meta", "tutor_role", fmt)
    if re.search(r"\b(confus\w*|unclear|don't understand|do not understand)\b", q):
        return Request("repair", "previous_response", fmt)
    if has_current_asset and re.search(r"line by line|each line", q):
        return Request("walkthrough", "current_asset", fmt)
    if fmt != "prose" or re.search(r"\bexample\b", q):
        return Request("create_example", "current_concept", fmt, another)
    if re.search(r"\b(simpl\w*|easier)\b", q):
        return Request("simplify", "previous_response", fmt)
    if re.search(r"\b(compar\w*|contrast|difference)\b", q):
        return Request("compare", "comparison", fmt)
    return Request("explain", "current_concept", fmt)

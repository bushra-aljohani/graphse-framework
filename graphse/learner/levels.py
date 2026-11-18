"""Coarse learner level used to condition the dialogue tutor's vocabulary and depth.

A deterministic heuristic over graded evidence, not a fitted model:
proficient  - at least 5 graded answers and average mastery >= 0.75
developing  - at least 2 graded answers, or average mastery >= 0.45
foundation  - otherwise
"""
from __future__ import annotations

LEVELS = ("foundation", "developing", "proficient")


def learner_level(graded_answers: int, mastery: dict[str, float], prior: float = 0.30) -> str:
    average = sum(mastery.values()) / len(mastery) if mastery else prior
    if graded_answers >= 5 and average >= 0.75:
        return "proficient"
    if graded_answers >= 2 or average >= 0.45:
        return "developing"
    return "foundation"

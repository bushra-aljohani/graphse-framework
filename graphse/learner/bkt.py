"""Standard Bayesian Knowledge Tracing (Corbett and Anderson, 1994).

After each graded answer the posterior is computed and then the learning
transition is applied, whether the answer was correct or not. "Unsure"
responses are not graded and do not update the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

RULE_VERSION = "bkt-standard-v1"


@dataclass(frozen=True)
class BKTParams:
    prior: float = 0.30        # P(L0)
    transit: float = 0.30      # P(T)
    guess: float = 0.20        # P(G)
    slip: float = 0.10         # P(S)
    mastery_threshold: float = 0.95


def update(p_known: float, correct: bool, params: BKTParams = BKTParams()) -> float:
    if correct:
        num = p_known * (1 - params.slip)
        posterior = num / (num + (1 - p_known) * params.guess)
    else:
        num = p_known * params.slip
        posterior = num / (num + (1 - p_known) * (1 - params.guess))
    return posterior + (1 - posterior) * params.transit


def is_mastered(p_known: float, params: BKTParams = BKTParams()) -> bool:
    return p_known >= params.mastery_threshold

"""Reciprocal-rank fusion (RRF) of several rankings.

RRF(d) = sum over rankings s of 1 / (k + rank_s(d)), using the top `depth`
results of each ranking. In the evaluated research variant, depth=20 and k=10
were selected on development data only.
"""
from __future__ import annotations


def rrf(rankings: list[list[tuple[str, float]]], depth: int = 20, k: int = 10) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking[:depth], start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

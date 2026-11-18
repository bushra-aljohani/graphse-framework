"""Deterministic next-step recommendations.

Inputs are graded evidence only: mastery, incorrect answers, open misconception
signals, recency, and structural learning-order predecessors. Structural order
is labelled as a proxy, never as a validated prerequisite.

All rules use one `attention_threshold` (by default the mastery threshold), so
a concept shown as "not mastered" can still be recommended.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    concept_id: str
    kind: str          # "learning_order", "misconception", "slipping", "weakness"
    priority: float
    reason: str


def recommend(mastery: dict[str, float], wrong: dict[str, int], open_misconceptions: dict[str, int],
              days_since_evidence: dict[str, int], predecessors: dict[str, list[str]],
              attention_threshold: float = 0.95, slipping_band: float = 0.20, stale_days: int = 7,
              prior: float = 0.30, limit: int = 5) -> list[Recommendation]:
    p = lambda c: mastery.get(c, prior)
    needs_attention = lambda c: p(c) < attention_threshold or wrong.get(c) or open_misconceptions.get(c)
    candidates = set(mastery) | set(wrong) | set(open_misconceptions)
    items: list[Recommendation] = []
    used: set[str] = set()

    for target in sorted(candidates, key=lambda c: (p(c), -wrong.get(c, 0))):       # 1. learning order
        if not needs_attention(target):
            continue
        weaker = [x for x in predecessors.get(target, []) if p(x) < p(target) and p(x) < attention_threshold]
        if weaker:
            pre = min(weaker, key=p)
            items.append(Recommendation(pre, "learning_order", 3 + (1 - p(pre)),
                                        f"appears before {target} in the source's learning order (proxy)"))
            used.add(pre)
            break

    for c, n in sorted(open_misconceptions.items(), key=lambda x: -x[1]):            # 2. misconceptions
        if c not in used and n:
            items.append(Recommendation(c, "misconception", 2.5 + (1 - p(c)), f"{n} open misconception signal(s)"))
            used.add(c)

    for c in sorted(mastery, key=lambda c: -p(c)):                                   # 3. slipping
        age = days_since_evidence.get(c)
        if c not in used and attention_threshold - slipping_band <= p(c) < attention_threshold and age is not None and age >= stale_days:
            items.append(Recommendation(c, "slipping", 2 + (1 - p(c)), f"near mastery, no graded evidence for {age} days"))
            used.add(c)

    for c in sorted(candidates, key=lambda c: (p(c), -wrong.get(c, 0))):             # 4. weakness
        if c not in used and needs_attention(c):
            items.append(Recommendation(c, "weakness", 1 + (1 - p(c)) + min(wrong.get(c, 0), 3) * 0.15,
                                        f"{wrong.get(c, 0)} incorrect answer(s), mastery {p(c):.2f}"))
            used.add(c)

    return sorted(items, key=lambda r: (-r.priority, r.concept_id))[:limit]

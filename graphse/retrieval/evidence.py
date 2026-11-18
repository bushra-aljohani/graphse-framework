"""Server-side evidence selection by provenance lookup.

For a concept that the learner or the policy has already chosen, the evidence
set is that concept's own verified source spans. Evidence supplied by a client
is never used.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..knowledge.graph import KnowledgeGraph


@dataclass(frozen=True)
class Citation:
    concept_id: str
    concept_name: str
    section_id: str
    page: int
    text: str


def citations_for_concept(kg: KnowledgeGraph, concept_id: str) -> list[Citation]:
    concept = kg.concepts.get(concept_id)
    if concept is None:
        return []
    out, seen = [], set()
    for p in concept.provenance:
        key = (p.chunk_id, p.span)
        if key not in seen:
            seen.add(key)
            out.append(Citation(concept_id, concept.name, p.section_id, p.page, p.span))
    return out


def numbered_excerpts(citations: list[Citation]) -> str:
    """The numbered evidence block given to agents; agents cite by index."""
    return "\n".join(f"[{i}] p.{c.page}: {c.text}" for i, c in enumerate(citations))

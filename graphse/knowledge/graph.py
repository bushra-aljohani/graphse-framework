"""In-memory knowledge graph and its construction pipeline.

Pipeline: proposals -> provenance gate -> canonicalisation -> evidence judge
(advisory) -> relation filter -> semantic edges; plus structural edges derived
from the document layer only.

Relation filter: only explicitly supported, judge-accepted `related_to` and
`improves` relations are loaded. All other relations (high-impact types such as
`prerequisite_of`, and inferred candidates) are excluded from the graph.

OPTIONAL, NOT USED IN THE PAPER: with `keep_review_queue=True` the excluded
relations are kept in `review_queue` for a human-review workflow. No human
review was performed or evaluated in the study described in the paper.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from .corpus import Corpus
from .provenance_gate import GateResult, run_gate
from .schema import SEMANTIC_RELATIONS, Concept, Edge, Proposal, Provenance


class EvidenceJudge(Protocol):
    """Advisory semantic check (in the reference system, an LLM from a different family)."""
    def supports(self, proposal: Proposal) -> bool: ...


class AcceptAllJudge:
    def supports(self, proposal: Proposal) -> bool:
        return True


@dataclass
class KnowledgeGraph:
    corpus: Corpus
    concepts: dict[str, Concept] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    excluded_relations: int = 0                                       # not loaded by the relation filter
    review_queue: list[Proposal] = field(default_factory=list)        # optional; see module docstring
    rejected: list[GateResult] = field(default_factory=list)

    # ----- queries used online -----
    def concept_by_name(self, name: str) -> Concept | None:
        key = name.casefold()
        return next((c for c in self.concepts.values() if c.name.casefold() == key), None)

    def neighbours(self, concept_id: str, edge_types: set[str] | None = None) -> list[tuple[str, Edge]]:
        out = []
        for e in self.edges:
            if edge_types and e.edge_type not in edge_types:
                continue
            if e.source == concept_id:
                out.append((e.target, e))
            elif e.target == concept_id:
                out.append((e.source, e))
        return out

    def predecessors(self, concept_id: str) -> list[str]:
        """Structural learning-order predecessors (a proxy, not a validated prerequisite)."""
        return [e.source for e in self.edges if e.edge_type == "SEQUENCE_PRECEDES" and e.target == concept_id]

    def counts(self) -> dict[str, int]:
        c = defaultdict(int)
        for e in self.edges:
            c[e.edge_type] += 1
        return {"concepts": len(self.concepts),
                "provenance": sum(len(x.provenance) for x in self.concepts.values()),
                **dict(c), "excluded_relations": self.excluded_relations, "rejected": len(self.rejected)}


def _provenance(p: Proposal, corpus: Corpus) -> Provenance:
    return Provenance(p.chunk_id, corpus.chunks[p.chunk_id].section_id, p.page, p.span,
                      p.support_status, float(p.confidence))


def build_graph(corpus: Corpus, proposals: list[Proposal], judge: EvidenceJudge | None = None,
                keep_review_queue: bool = False) -> KnowledgeGraph:
    judge = judge or AcceptAllJudge()
    kg = KnowledgeGraph(corpus)
    passed, kg.rejected = run_gate(proposals, corpus.chunks)

    # Canonicalisation: one concept per case-insensitive name; every merged
    # proposal keeps its provenance, and a name extracted under several types
    # keeps all of them.
    by_name: dict[str, Concept] = {}
    for p in (x for x in passed if x.kind == "node" and judge.supports(x)):
        key = p.name.casefold()
        if key not in by_name:
            by_name[key] = Concept(f"C{len(by_name) + 1:04d}", p.name.strip(), [p.node_type])
        concept = by_name[key]
        if p.node_type not in concept.types:
            concept.types.append(p.node_type)
        concept.provenance.append(_provenance(p, corpus))
    kg.concepts = {c.concept_id: c for c in by_name.values()}

    # Relation filter: explicitly supported, judge-accepted ordinary relations
    # enter the graph; high-impact or inferred relations are excluded (and only
    # queued for review when the optional review queue is enabled).
    seen = set()
    for p in (x for x in passed if x.kind == "relation"):
        src, tgt = by_name.get(p.source.casefold()), by_name.get(p.target.casefold())
        if src is None or tgt is None:          # endpoint removed by the judge
            continue
        if p.relation_type in SEMANTIC_RELATIONS and p.support_status == "EXPLICIT" and judge.supports(p):
            key = (src.concept_id, p.relation_type, tgt.concept_id, p.page, p.span)
            if key not in seen:
                seen.add(key)
                kg.edges.append(Edge(src.concept_id, tgt.concept_id, p.relation_type.upper(),
                                     "text_supported", _provenance(p, corpus)))
        else:
            kg.excluded_relations += 1
            if keep_review_queue:
                kg.review_queue.append(p)

    kg.edges.extend(structural_edges(kg))
    return kg


def structural_edges(kg: KnowledgeGraph) -> list[Edge]:
    """Edges derived from document organisation only; they assert location, not dependency."""
    by_leaf: dict[str, set[str]] = defaultdict(set)
    for c in kg.concepts.values():
        for prov in c.provenance:
            by_leaf[prov.section_id].add(c.concept_id)
    edges, seen = [], set()
    for members in by_leaf.values():
        for a, b in combinations(sorted(members), 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append(Edge(a, b, "CO_LOCATED", "structural"))
    for first, second in kg.corpus.adjacent_leaf_pairs():
        for a in sorted(by_leaf.get(first, ())):
            for b in sorted(by_leaf.get(second, ())):
                if a != b and ("seq", a, b) not in seen:
                    seen.add(("seq", a, b))
                    edges.append(Edge(a, b, "SEQUENCE_PRECEDES", "structural"))
    return edges

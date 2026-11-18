"""Embedding-and-graph retriever (four deterministic stages after the query embedding).

1. Link: the query is linked to the closest concepts by cosine similarity.
2. Traverse: typed edges are followed for a bounded number of hops.
3. Re-score and prune: touched concepts are scored by seed similarity, edge
   weight and hop decay; low-scoring ones are dropped.
4. Load provenance: the retained concepts' source spans become citations.

The weights below are illustrative defaults for the abstraction, not tuned values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..knowledge.graph import KnowledgeGraph
from .embed import Embedder, cosine
from .evidence import Citation, citations_for_concept

EDGE_WEIGHT = {"RELATED_TO": 1.0, "IMPROVES": 1.0, "CO_LOCATED": 0.5, "SEQUENCE_PRECEDES": 0.4}


@dataclass
class RetrievalResult:
    ranking: list[tuple[str, float]]               # (concept_id, score), best first
    citations: list[Citation]
    stages: dict = field(default_factory=dict)     # per-stage trace for auditing


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph, embedder: Embedder, hop_decay: float = 0.6):
        self.kg, self.embedder, self.hop_decay = kg, embedder, hop_decay
        self.vectors = {cid: embedder.embed(self._text(c)) for cid, c in kg.concepts.items()}

    @staticmethod
    def _text(concept) -> str:
        return concept.name + " " + " ".join(p.span for p in concept.provenance)

    def similarities(self, query: str) -> list[tuple[str, float]]:
        q = self.embedder.embed(query)
        return sorted(((cid, cosine(q, v)) for cid, v in self.vectors.items()), key=lambda x: (-x[1], x[0]))

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.40,
                 max_hops: int = 2, max_nodes: int = 30, prune_below: float = 0.05) -> RetrievalResult:
        sims = self.similarities(query)
        seeds = [(cid, s) for cid, s in sims if s >= min_score][:top_k]                 # stage 1
        scores = {cid: s for cid, s in seeds}
        frontier = list(seeds)
        for hop in range(1, max_hops + 1):                                              # stage 2
            nxt = []
            for cid, base in frontier:
                for other, edge in self.kg.neighbours(cid):
                    s = base * EDGE_WEIGHT.get(edge.edge_type, 0.3) * self.hop_decay ** hop   # stage 3
                    is_new = other not in scores
                    if is_new and len(scores) >= max_nodes:       # keep the subgraph bounded
                        continue
                    if is_new or s > scores[other]:
                        scores[other] = s
                        nxt.append((other, s))
            frontier = nxt
        ranking = sorted(((c, s) for c, s in scores.items() if s >= prune_below), key=lambda x: (-x[1], x[0]))
        citations = [c for cid, _ in ranking for c in citations_for_concept(self.kg, cid)]   # stage 4
        return RetrievalResult(ranking, citations,
                               {"seeds": seeds, "touched": len(scores), "kept": len(ranking)})

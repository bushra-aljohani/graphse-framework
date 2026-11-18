"""Data model of the two-layer knowledge graph.

Layer A (document layer) mirrors the organisation of the source corpus:
areas, topics and (possibly nested) subtopics, each leaf holding text chunks.
Layer B (concept layer) holds concepts proposed by an extraction model. Every
concept and semantic relation keeps the verbatim source span it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NODE_TYPES = {"Concept", "Definition", "Principle", "Technique", "Artifact"}
SEMANTIC_RELATIONS = {"related_to", "improves"}
# Relation types with pedagogical consequences are never loaded into the graph.
HIGH_IMPACT_RELATIONS = {"prerequisite_of", "contrasts_with", "assessed_by"}
RELATION_TYPES = SEMANTIC_RELATIONS | HIGH_IMPACT_RELATIONS
SUPPORT_STATUS = {"EXPLICIT", "INFERRED_CANDIDATE"}
STRUCTURAL_RELATIONS = {"CO_LOCATED", "SEQUENCE_PRECEDES"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str          # leaf section (Layer A) the chunk belongs to
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class Provenance:
    chunk_id: str
    section_id: str
    page: int
    span: str                # verbatim text from the chunk
    support_status: str
    confidence: float


@dataclass
class Proposal:
    """One item proposed by the extraction model for one chunk (not yet trusted)."""
    kind: str                # "node" or "relation"
    chunk_id: str
    page: int
    span: str
    support_status: str
    confidence: float
    name: str | None = None          # node
    node_type: str | None = None     # node
    source: str | None = None        # relation: source concept name
    target: str | None = None        # relation: target concept name
    relation_type: str | None = None


@dataclass
class Concept:
    concept_id: str
    name: str
    types: list[str]
    provenance: list[Provenance] = field(default_factory=list)

    @property
    def primary_type(self) -> str:
        return self.types[0]


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    edge_type: str           # RELATED_TO, IMPROVES, CO_LOCATED, SEQUENCE_PRECEDES
    derivation: str          # "text_supported" or "structural"
    evidence: Provenance | None = None

"""Deterministic provenance gate for extraction proposals.

Passing the gate means a proposal is *traceable*: it follows the schema and its
quoted span occurs verbatim in the referenced chunk, on a page inside that chunk.
It does not mean the proposal is semantically correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import NODE_TYPES, RELATION_TYPES, SUPPORT_STATUS, Chunk, Proposal


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class GateResult:
    proposal: Proposal
    passed: bool
    reason: str | None = None


def check_proposal(p: Proposal, chunks: dict[str, Chunk], accepted_names: set[str] | None = None) -> GateResult:
    """Return why a proposal fails, or a pass.

    accepted_names: casefolded names of nodes that already passed; relation
    endpoints must be among them.
    """
    if p.kind not in {"node", "relation"}:
        return GateResult(p, False, "unknown_kind")
    if p.support_status not in SUPPORT_STATUS:
        return GateResult(p, False, "invalid_support_status")
    if not isinstance(p.confidence, (int, float)) or not 0.0 <= p.confidence <= 1.0:
        return GateResult(p, False, "confidence_out_of_range")
    if p.kind == "node":
        if p.node_type not in NODE_TYPES:
            return GateResult(p, False, "unsupported_node_type")
        if not p.name or not p.name.strip():
            return GateResult(p, False, "empty_name")
    else:
        if p.relation_type not in RELATION_TYPES:
            return GateResult(p, False, "unsupported_relation_type")
        if not p.source or not p.target:
            return GateResult(p, False, "missing_endpoint")
        names = accepted_names or set()
        if p.source.casefold() not in names:
            return GateResult(p, False, "source_endpoint_not_accepted")
        if p.target.casefold() not in names:
            return GateResult(p, False, "target_endpoint_not_accepted")
    chunk = chunks.get(p.chunk_id)
    if chunk is None:
        return GateResult(p, False, "unknown_chunk")
    if not isinstance(p.page, int) or not chunk.page_start <= p.page <= chunk.page_end:
        return GateResult(p, False, "page_outside_chunk")
    if not p.span or _normalise(p.span) not in _normalise(chunk.text):
        return GateResult(p, False, "span_not_verbatim")
    return GateResult(p, True)


def run_gate(proposals: list[Proposal], chunks: dict[str, Chunk]) -> tuple[list[Proposal], list[GateResult]]:
    """Gate nodes first, then relations against the accepted node names."""
    passed, rejected = [], []
    for p in (x for x in proposals if x.kind != "relation"):
        r = check_proposal(p, chunks)
        (passed if r.passed else rejected).append(p if r.passed else r)
    names = {p.name.casefold() for p in passed}
    for p in (x for x in proposals if x.kind == "relation"):
        r = check_proposal(p, chunks, names)
        (passed if r.passed else rejected).append(p if r.passed else r)
    return passed, rejected

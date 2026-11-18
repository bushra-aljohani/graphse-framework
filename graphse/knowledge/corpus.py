"""Loading the document layer (Layer A) from a structured corpus description."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Chunk, Proposal


@dataclass
class Section:
    section_id: str
    title: str
    kind: str                       # "area", "topic" or "subtopic"
    parent_id: str | None
    children: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    sections: dict[str, Section]
    chunks: dict[str, Chunk]

    def leaves(self) -> list[str]:
        return [s.section_id for s in self.sections.values() if s.kind != "area" and not s.children]

    def ancestors(self, section_id: str) -> list[Section]:
        out, current = [], self.sections[section_id].parent_id
        while current:
            out.append(self.sections[current])
            current = self.sections[current].parent_id
        return out

    def adjacent_leaf_pairs(self) -> list[tuple[str, str]]:
        """Pairs (a, b) of sibling leaves where b directly follows a in reading order."""
        pairs = []
        for section in self.sections.values():
            leaves = [c for c in section.children if not self.sections[c].children]
            pairs.extend(zip(leaves, leaves[1:]))
        return pairs


def load_corpus(path: str | Path) -> Corpus:
    """Read a corpus file with nested areas/topics/subtopics and leaf chunks.

    Format: {"areas": [{"id", "title", "topics": [{"id", "title", "subtopics": [
              {"id", "title", "subtopics": [...], "chunks": [{"id", "page_start", "page_end", "text"}]}]}]}]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sections: dict[str, Section] = {}
    chunks: dict[str, Chunk] = {}

    def add(node: dict, kind: str, parent: str | None) -> None:
        sections[node["id"]] = Section(node["id"], node["title"], kind, parent)
        if parent:
            sections[parent].children.append(node["id"])
        for child in node.get("topics", []):
            add(child, "topic", node["id"])
        for child in node.get("subtopics", []):
            add(child, "subtopic", node["id"])
        for c in node.get("chunks", []):
            chunks[c["id"]] = Chunk(c["id"], node["id"], c["page_start"], c["page_end"], c["text"])

    for area in data["areas"]:
        add(area, "area", None)
    return Corpus(sections, chunks)


def load_proposals(path: str | Path) -> list[Proposal]:
    """Read extraction proposals ({"proposals": [...]}) produced by an extraction model."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Proposal(**p) for p in data["proposals"]]

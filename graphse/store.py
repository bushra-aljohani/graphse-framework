"""Minimal in-memory learner store with all-or-nothing transactions.

The reference deployment uses a relational database. What matters for the
architecture is that grading, the mastery update and the stored result are
written in ONE transaction, so a retry after a failure cannot count an answer
twice.
"""
from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LearnerStore:
    mastery: dict[tuple[str, str], float] = field(default_factory=dict)
    mastery_log: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    attempts: dict[str, dict] = field(default_factory=dict)
    sessions: dict[str, dict] = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    misconceptions: list[dict] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    _STATE = ("mastery", "mastery_log", "interactions", "attempts", "sessions", "turns", "misconceptions")

    @contextmanager
    def transaction(self):
        with self._lock:
            snapshot = {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}
            try:
                yield self
            except BaseException:
                for k, v in snapshot.items():
                    setattr(self, k, v)
                raise

    def student_mastery(self, student_id: str) -> dict[str, float]:
        return {c: p for (s, c), p in self.mastery.items() if s == student_id}

    def graded_count(self, student_id: str) -> int:
        return sum(1 for i in self.interactions if i["student_id"] == student_id and i["label"] != "unsure")

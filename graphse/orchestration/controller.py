"""Learning controller: deterministic orchestration of the agents.

The controller never writes instructional content. For each request it checks
ownership, selects evidence on the server, attaches the learner level, calls
one agent through the harness, records the turn and enforces the session state
machine. Agents never call one another, and conversation never changes mastery.

Session states: learning -> assessment_pending -> assessment_in_progress -> completed
"""
from __future__ import annotations

import uuid
from collections import Counter

from ..agents.mcq_author import MCQAuthor
from ..agents.tutor import DialogueTutor, InitialTutor
from ..assessment.grading import AssessmentService, ItemBank
from ..knowledge.graph import KnowledgeGraph
from ..learner.bkt import BKTParams
from ..learner.levels import learner_level
from ..learner.recommendations import recommend
from ..retrieval.evidence import citations_for_concept
from ..store import LearnerStore


class StateError(RuntimeError):
    """An operation does not fit the session's current state."""


class LearningController:
    def __init__(self, kg: KnowledgeGraph, store: LearnerStore, initial: InitialTutor, dialogue: DialogueTutor,
                 author: MCQAuthor, bank: ItemBank, assessment: AssessmentService, params: BKTParams = BKTParams()):
        self.kg, self.store, self.initial, self.dialogue = kg, store, initial, dialogue
        self.author, self.bank, self.assessment, self.params = author, bank, assessment, params

    # ----- sessions -----
    def start_session(self, student_id: str) -> str:
        sid = uuid.uuid4().hex
        with self.store.transaction() as s:
            s.sessions[sid] = {"student_id": student_id, "status": "learning", "concepts": [], "turns": 0,
                               "attempts": []}
        return sid

    def _session(self, student_id: str, sid: str) -> dict:
        session = self.store.sessions.get(sid)
        if session is None or session["student_id"] != student_id:   # ownership is checked on every access
            raise PermissionError("session not found for this student")
        return session

    def _record_turn(self, student_id: str, sid: str, concept_id: str, role: str, payload: dict) -> None:
        with self.store.transaction() as s:
            session = s.sessions[sid]
            if session["status"] != "learning":
                raise StateError("the learning phase of this session is closed")
            if concept_id not in session["concepts"]:
                session["concepts"].append(concept_id)
            session["turns"] += 1
            s.turns.append({"student_id": student_id, "session_id": sid, "concept_id": concept_id,
                            "role": role, **payload})

    # ----- teaching (no mastery change) -----
    def explain(self, student_id: str, sid: str, concept_id: str) -> dict:
        session = self._session(student_id, sid)
        if session["status"] != "learning":
            raise StateError("the learning phase of this session is closed")
        concept = self.kg.concepts[concept_id]
        citations = citations_for_concept(self.kg, concept_id)
        p = self.store.mastery.get((student_id, concept_id), self.params.prior)
        lesson = self.initial.lesson(concept.name, p, citations)
        self._record_turn(student_id, sid, concept_id, "initial_lesson", {"grounded": lesson["grounded"]})
        return lesson

    def chat(self, student_id: str, sid: str, concept_id: str, question: str,
             client_citations: list | None = None) -> dict:
        """`client_citations` is accepted only to show it is ignored: evidence comes from the graph."""
        session = self._session(student_id, sid)
        if session["status"] != "learning":
            raise StateError("the learning phase of this session is closed")
        citations = citations_for_concept(self.kg, concept_id)
        level = learner_level(self.store.graded_count(student_id), self.store.student_mastery(student_id),
                              self.params.prior)
        result = self.dialogue.answer(question, self.kg.concepts[concept_id].name, citations, level)
        self._record_turn(student_id, sid, concept_id, "dialogue",
                          {"response_type": result["response_type"], "outcome": result.get("outcome")})
        if result.get("misconception"):
            with self.store.transaction() as s:
                s.misconceptions.append({"student_id": student_id, "concept_id": concept_id, "status": "open",
                                         "method": "tutor_detected_claim", **result["misconception"]})
        return {**result, "learner_level": level}

    # ----- assessment -----
    def finish_session(self, student_id: str, sid: str) -> None:
        session = self._session(student_id, sid)
        if session["status"] != "learning" or session["turns"] == 0:
            raise StateError("nothing to assess yet")
        with self.store.transaction() as s:
            s.sessions[sid]["status"] = "assessment_pending"

    def start_test(self, student_id: str, sid: str) -> list[dict]:
        session = self._session(student_id, sid)
        if session["status"] != "assessment_pending":
            raise StateError("the session is not waiting for an assessment")
        # Write every item first (stored-first, else author), so a rejected item leaves no partial test.
        items = []
        for cid in session["concepts"]:
            stored = self.bank.servable(cid)
            items.append((cid, stored[0] if stored else None))
        items = [(cid, item or self.author.write(self.kg.concepts[cid].name, citations_for_concept(self.kg, cid)))
                 for cid, item in items]
        public = []
        with self.store.transaction() as s:
            for cid, item in items:
                if not self.bank.servable(cid):
                    self.bank.add(cid, item)
                view = self.assessment.start_attempt(student_id, cid, item, run_id=sid)
                s.sessions[sid]["attempts"].append(view["attempt_id"])
                public.append(view)
            s.sessions[sid]["status"] = "assessment_in_progress"
        return public

    def complete_test(self, student_id: str, sid: str) -> dict:
        session = self._session(student_id, sid)
        if session["status"] != "assessment_in_progress":
            raise StateError("no assessment in progress")
        results = [self.store.attempts[a]["result"] for a in session["attempts"]]
        if any(r is None for r in results):
            raise StateError("every question must be answered first")
        with self.store.transaction() as s:
            s.sessions[sid]["status"] = "completed"
        correct = sum(r["label"] == "correct" for r in results)
        return {"correct": correct, "total": len(results), "score": correct / len(results)}

    # ----- recommendations -----
    def recommendations(self, student_id: str, days_since_evidence: dict[str, int] | None = None):
        mastery = self.store.student_mastery(student_id)
        wrong = Counter(i["concept_id"] for i in self.store.interactions
                        if i["student_id"] == student_id and i["label"] == "incorrect")
        open_mis = Counter(m["concept_id"] for m in self.store.misconceptions
                           if m["student_id"] == student_id and m["status"] == "open")
        preds = {c: self.kg.predecessors(c) for c in set(mastery) | set(wrong) | set(open_mis)}
        return recommend(mastery, dict(wrong), dict(open_mis), days_since_evidence or {}, preds,
                         attention_threshold=self.params.mastery_threshold, prior=self.params.prior)

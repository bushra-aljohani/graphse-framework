"""Assessment delivery and deterministic grading.

* Options are shuffled per attempt; the answer key never leaves the server
  before submission.
* Grading compares the selected option with the stored key; no model decides
  correctness.
* A repeated submission of the same attempt returns the stored result.
* Grading, the BKT update and the stored result are one transaction.
"""
from __future__ import annotations

import random
import uuid

from ..learner.bkt import RULE_VERSION, BKTParams, update
from ..store import LearnerStore


class ItemBank:
    """Stored-first item selection; items reserved for pre/post-tests are never served as practice."""

    def __init__(self):
        self.items: dict[str, list[dict]] = {}
        self.reserved: set[str] = set()

    def add(self, concept_id: str, item: dict, reserved: bool = False) -> str:
        item_id = item.get("item_id") or f"item-{uuid.uuid4().hex[:8]}"
        self.items.setdefault(concept_id, []).append({**item, "item_id": item_id})
        if reserved:
            self.reserved.add(item_id)
        return item_id

    def servable(self, concept_id: str) -> list[dict]:
        return [i for i in self.items.get(concept_id, []) if i["item_id"] not in self.reserved]


class AssessmentService:
    def __init__(self, store: LearnerStore, params: BKTParams = BKTParams(), seed: int | None = None):
        self.store, self.params, self.rng = store, params, random.Random(seed)

    def start_attempt(self, student_id: str, concept_id: str, item: dict, run_id: str | None = None) -> dict:
        order = list(range(len(item["options"])))
        self.rng.shuffle(order)
        attempt_id = uuid.uuid4().hex
        with self.store.transaction() as s:
            s.attempts[attempt_id] = {"student_id": student_id, "concept_id": concept_id, "item": item,
                                      "order": order, "run_id": run_id, "result": None}
        return {"attempt_id": attempt_id, "concept_id": concept_id, "question": item["question"],
                "options": [{"id": str(i), "text": item["options"][o]} for i, o in enumerate(order)]}

    def answer(self, student_id: str, attempt_id: str, option_id: str | None = None, unsure: bool = False) -> dict:
        with self.store.transaction() as s:
            attempt = s.attempts.get(attempt_id)
            if attempt is None or attempt["student_id"] != student_id:
                raise KeyError("attempt not found for this student")
            if attempt["result"] is not None:
                return attempt["result"]                       # replay, no second update
            item, cid, key = attempt["item"], attempt["concept_id"], (student_id, attempt["concept_id"])
            if unsure:
                result = {"label": "unsure", "p_before": s.mastery.get(key, self.params.prior),
                          "p_after": s.mastery.get(key, self.params.prior)}
            else:
                valid = [str(i) for i in range(len(attempt["order"]))]
                if option_id not in valid:
                    raise ValueError("select one of the supplied options")
                correct = attempt["order"][int(option_id)] == item["correct_index"]
                p_before = s.mastery.get(key, self.params.prior)
                p_after = update(p_before, correct, self.params)
                s.mastery[key] = p_after
                s.mastery_log.append({"student_id": student_id, "concept_id": cid, "p_before": p_before,
                                      "p_after": p_after, "rule_version": RULE_VERSION, "attempt_id": attempt_id})
                result = {"label": "correct" if correct else "incorrect", "p_before": p_before, "p_after": p_after}
                if not correct:
                    s.misconceptions.append({"student_id": student_id, "concept_id": cid, "status": "open",
                                             "method": "objective_wrong_answer", "attempt_id": attempt_id})
            s.interactions.append({"student_id": student_id, "concept_id": cid, "label": result["label"],
                                   "attempt_id": attempt_id})
            result.update(explanation=item["explanation"],
                          correct_option=str(attempt["order"].index(item["correct_index"])))
            attempt["result"] = result
            return result

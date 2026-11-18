import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from graphse.agents import DialogueTutor, InitialTutor, MCQAuthor, ScriptedLLM
from graphse.assessment import AssessmentService, ItemBank
from graphse.orchestration import LearningController, StateError
from graphse.store import LearnerStore

ITEM = {"question": "What does coupling measure?", "options": ["Dependence", "Speed", "Size"],
        "correct_index": 0, "explanation": "It measures dependence.", "support_quote": "q"}


def option(view, text):
    return next(o["id"] for o in view["options"] if o["text"] == text)


def test_key_hidden_and_grading_deterministic():
    store = LearnerStore(); svc = AssessmentService(store, seed=3)
    view = svc.start_attempt("s1", "C1", ITEM)
    assert "correct_index" not in json.dumps(view)
    r = svc.answer("s1", view["attempt_id"], option(view, "Speed"))
    assert r["label"] == "incorrect" and abs(r["p_after"] - 0.33559322033898303) < 1e-12


def test_invalid_option_unsure_and_ownership():
    store = LearnerStore(); svc = AssessmentService(store, seed=3)
    view = svc.start_attempt("s1", "C1", ITEM)
    with pytest.raises(ValueError):
        svc.answer("s1", view["attempt_id"], "3")
    with pytest.raises(KeyError):
        svc.answer("s2", view["attempt_id"], "0")
    assert svc.answer("s1", view["attempt_id"], unsure=True)["label"] == "unsure" and store.mastery_log == []


def test_replay_counts_once():
    store = LearnerStore(); svc = AssessmentService(store, seed=3)
    view = svc.start_attempt("s1", "C1", ITEM)
    first = svc.answer("s1", view["attempt_id"], option(view, "Dependence"))
    assert svc.answer("s1", view["attempt_id"], option(view, "Dependence")) == first
    assert len(store.mastery_log) == 1


class FailOnce(list):
    def __init__(self):
        super().__init__(); self.failed = False

    def append(self, x):
        if not self.failed:
            self.failed = True
            raise OSError("injected storage failure after grading")
        super().append(x)


def test_interrupted_answer_is_not_counted_twice():
    store = LearnerStore(); svc = AssessmentService(store, seed=3)
    view = svc.start_attempt("s1", "C1", ITEM)
    store.interactions = FailOnce()
    with pytest.raises(OSError):
        svc.answer("s1", view["attempt_id"], option(view, "Dependence"))
    assert store.mastery_log == [] and ("s1", "C1") not in store.mastery     # rolled back
    store.interactions = list(store.interactions)
    svc.answer("s1", view["attempt_id"], option(view, "Dependence"))
    assert len(store.mastery_log) == 1


def test_simultaneous_submissions_count_once():
    store = LearnerStore(); svc = AssessmentService(store, seed=3)
    view = svc.start_attempt("s1", "C1", ITEM)
    gate = threading.Barrier(2)

    def submit():
        gate.wait()
        return svc.answer("s1", view["attempt_id"], option(view, "Dependence"))
    with ThreadPoolExecutor(2) as pool:
        a, b = [f.result() for f in [pool.submit(submit) for _ in range(2)]]
    assert a == b and len(store.mastery_log) == 1


def controller(kg, responses):
    store = LearnerStore(); llm = ScriptedLLM(responses)
    return LearningController(kg, store, InitialTutor(llm), DialogueTutor(llm), MCQAuthor(llm), ItemBank(),
                              AssessmentService(store, seed=1)), store, llm


def test_session_ownership(kg):
    ctl, _, _ = controller(kg, [])
    sid = ctl.start_session("alice")
    with pytest.raises(PermissionError):
        ctl.explain("bob", sid, kg.concept_by_name("Coupling").concept_id)


def test_state_machine_and_no_mastery_from_conversation(kg):
    cid = kg.concept_by_name("Coupling").concept_id
    ctl, store, llm = controller(kg, [{"answer": "a", "citation_refs": [0], "asset": None, "misconception": None}])
    sid = ctl.start_session("alice")
    with pytest.raises(StateError):
        ctl.finish_session("alice", sid)                      # nothing learned yet
    ctl.chat("alice", sid, cid, "What is coupling?", client_citations=[{"text": "UNTRUSTED"}])
    assert "UNTRUSTED" not in json.dumps(llm.calls) and store.mastery_log == []
    ctl.finish_session("alice", sid)
    with pytest.raises(StateError):
        ctl.chat("alice", sid, cid, "late question")           # learning phase closed


def test_reserved_items_are_never_served(kg):
    cid = kg.concept_by_name("Coupling").concept_id
    ctl, _, llm = controller(kg, [{"answer": "a", "citation_refs": [0], "asset": None, "misconception": None}])
    ctl.bank.add(cid, {**ITEM, "question": "reserved"}, reserved=True)
    ctl.bank.add(cid, {**ITEM, "question": "practice"})
    sid = ctl.start_session("alice")
    ctl.chat("alice", sid, cid, "What is coupling?")
    ctl.finish_session("alice", sid)
    [view] = ctl.start_test("alice", sid)
    assert view["question"] == "practice" and len(llm.calls) == 1   # stored item used, no authoring call
    with pytest.raises(StateError):
        ctl.complete_test("alice", sid)                        # unanswered

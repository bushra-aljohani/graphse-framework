import pytest

from graphse.agents import (DialogueTutor, GenerationRejected, InitialTutor, MCQAuthor, ProviderError, ScriptedLLM,
                            classify_request)
from graphse.retrieval import Citation

CITES = [Citation("C1", "Coupling", "A1.1", 10, "Coupling measures how strongly modules depend on each other.")]


def reply(refs=(0,), asset=None, misconception=None, answer="Coupling is dependence between modules."):
    return {"answer": answer, "citation_refs": list(refs), "asset": asset, "misconception": misconception}


def test_classification():
    assert classify_request("Show me a code example").output_format == "code"
    assert classify_request("I am confused").action == "repair"
    assert classify_request("Compare coupling and cohesion").action == "compare"


def test_no_evidence_means_no_model_call():
    llm = ScriptedLLM()
    r = DialogueTutor(llm).answer("What is coupling?", "Coupling", [], "foundation")
    assert r["response_type"] == "insufficient_evidence" and llm.calls == []
    assert InitialTutor(llm).lesson("Coupling", 0.3, [])["grounded"] is False and llm.calls == []


def test_first_pass_and_citation_filtering():
    llm = ScriptedLLM([reply(refs=[0, 5, -1])])
    r = DialogueTutor(llm).answer("What is coupling?", "Coupling", CITES, "foundation")
    assert r["outcome"] == "first_pass" and r["citation_refs"] == [0]


def test_invalid_json_is_repaired():
    llm = ScriptedLLM(["{not json", reply()])
    r = DialogueTutor(llm).answer("What is coupling?", "Coupling", CITES, "developing")
    assert r["outcome"] == "repaired" and r["model_calls"] == 2


def test_missing_citations_exhaust_into_fallback():
    llm = ScriptedLLM([reply(refs=[])] * 3)
    r = DialogueTutor(llm).answer("Explain coupling", "Coupling", CITES, "developing")
    assert r["outcome"] == "fallback" and r["response_type"] == "repair_fallback" and r["model_calls"] == 3


def test_requested_code_asset_is_enforced():
    code = {"type": "code", "content": "a = 1\nassert a == 1", "provenance": "generated_illustration"}
    llm = ScriptedLLM([reply(), reply(asset=code)])
    r = DialogueTutor(llm).answer("Give me a small code example", "Coupling", CITES, "developing")
    assert r["outcome"] == "repaired" and r["asset"]["type"] == "code"


def test_provider_failure_is_explicit():
    with pytest.raises(ProviderError):
        DialogueTutor(ScriptedLLM(["PROVIDER_ERROR"])).answer("What is coupling?", "Coupling", CITES, "foundation")


def test_misconception_needs_citation_and_confidence():
    m = {"label": "concept_confusion", "student_claim": "x", "correction": "y", "confidence": 0.6}
    r = DialogueTutor(ScriptedLLM([reply(misconception=m)])).answer("Is coupling cohesion?", "Coupling", CITES, "foundation")
    assert r["misconception"] is None
    m["confidence"] = 0.9
    r = DialogueTutor(ScriptedLLM([reply(misconception=m)])).answer("Is coupling cohesion?", "Coupling", CITES, "foundation")
    assert r["misconception"]["label"] == "concept_confusion"


def test_initial_tutor_retries_once_for_citations():
    llm = ScriptedLLM([{"explanation": "e", "example": "", "citation_refs": []},
                       {"explanation": "e", "example": "", "citation_refs": [0]}])
    lesson = InitialTutor(llm).lesson("Coupling", 0.3, CITES)
    assert lesson["grounded"] and lesson["model_calls"] == 2


def mcq(**over):
    item = {"question": "What does coupling measure?", "options": ["Dependence", "Speed", "Size"],
            "correct_index": 0, "explanation": "It measures dependence.", "support_ref": 0}
    return {**item, **over}


def test_mcq_server_attaches_quote():
    item = MCQAuthor(ScriptedLLM([mcq(support_quote="fabricated")])).write("Coupling", CITES)
    assert item["support_quote"] == CITES[0].text


def test_mcq_rejected_after_bounded_attempts():
    with pytest.raises(GenerationRejected) as exc:
        MCQAuthor(ScriptedLLM([mcq(support_ref=4), mcq(options=["A", "A", "B"]), mcq(correct_index=7)])).write("Coupling", CITES)
    assert len(exc.value.reasons) == 3


def test_mcq_without_evidence_makes_no_call():
    llm = ScriptedLLM()
    with pytest.raises(GenerationRejected):
        MCQAuthor(llm).write("Coupling", [])
    assert llm.calls == []

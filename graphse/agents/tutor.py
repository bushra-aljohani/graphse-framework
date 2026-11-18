"""The two tutoring agents.

InitialTutor writes the first lesson for a concept; DialogueTutor answers
follow-up requests. Both receive only server-selected, numbered evidence and
cite it by index. Neither changes the learner's mastery: only graded answers do.
"""
from __future__ import annotations

import json

from ..retrieval.evidence import Citation, numbered_excerpts
from .harness import HarnessResult, run_harness
from .llm import LLMClient
from .requests import Request, classify_request

INITIAL_SYSTEM = (
    "You are a software engineering tutor. Use ONLY the numbered excerpts. If they are too thin, say so. "
    'Return JSON: {"explanation": str, "example": str, "citation_refs": [int]} listing every excerpt used.'
)
DIALOGUE_SYSTEM = (
    "You are a software engineering tutor for one current concept. Obey the resolved request "
    "(action, target, output format). The learner level changes vocabulary and depth only and never "
    "overrides an explicit request. Ground factual claims in the numbered excerpts and cite them. "
    "Examples, code, tables and diagrams are generated illustrations and must be labelled as such. "
    'Return JSON: {"answer": str, "citation_refs": [int], "asset": null | '
    '{"type": "code"|"diagram"|"table", "content": str, "provenance": "generated_illustration"}, '
    '"misconception": null | {"label": str, "student_claim": str, "correction": str, "confidence": float}}'
)
NEEDS_CITATION = {"explain", "simplify", "compare", "repair"}
ASSET_FOR_FORMAT = {"code": "code", "diagram": "diagram", "table": "table"}
MISCONCEPTION_LABELS = {"concept_confusion", "incomplete_definition", "reversed_relationship",
                        "overgeneralization", "incorrect_application", "terminology_confusion"}
MIN_MISCONCEPTION_CONFIDENCE = 0.70


def valid_refs(refs: object, n: int) -> list[int]:
    out: list[int] = []
    for i in refs if isinstance(refs, list) else []:
        if type(i) is int and 0 <= i < n and i not in out:
            out.append(i)
    return out


class InitialTutor:
    def __init__(self, llm: LLMClient, temperature: float = 0.2):
        self.llm, self.temperature = llm, temperature

    def lesson(self, concept_name: str, mastery: float, citations: list[Citation]) -> dict:
        if not citations:
            return {"explanation": f"No source evidence is available for {concept_name} yet.",
                    "example": "", "citation_refs": [], "grounded": False, "model_calls": 0}
        scaffold = ("Mastery is low: give a worked example at a gentle, step-by-step pace." if mastery < 0.5 else
                    "Mastery is high: skip the basics; give a contrast or a transfer example.")
        prompt = f"Concept: {concept_name}\nMastery estimate: {mastery:.2f}\n{scaffold}\n\nExcerpts:\n{numbered_excerpts(citations)}"
        calls, lesson = 0, {}
        for reminder in ("", "\nREMINDER: citation_refs must not be empty."):
            calls += 1
            try:
                lesson = json.loads(self.llm.complete(INITIAL_SYSTEM, [{"role": "user", "content": prompt + reminder}],
                                                      self.temperature))
            except json.JSONDecodeError:
                lesson = {}
            if valid_refs(lesson.get("citation_refs"), len(citations)):
                break
        refs = valid_refs(lesson.get("citation_refs"), len(citations))
        return {"explanation": lesson.get("explanation", ""), "example": lesson.get("example", ""),
                "citation_refs": refs, "citations_used": [citations[i] for i in refs],
                "grounded": bool(refs), "model_calls": calls}


class DialogueTutor:
    def __init__(self, llm: LLMClient, max_attempts: int = 3):
        self.llm, self.max_attempts = llm, max_attempts

    def answer(self, question: str, concept_name: str, citations: list[Citation], learner_level: str,
               history: list[dict] | None = None, current_asset: dict | None = None, log=None) -> dict:
        request = classify_request(question, current_asset is not None)
        if request.action != "meta" and not citations and current_asset is None:
            return {"answer": "I do not have source evidence for this concept yet.", "citation_refs": [],
                    "response_type": "insufficient_evidence", "action": request.action, "model_calls": 0}
        prompt = (f"Concept: {concept_name}\nLearner level: {learner_level}\n"
                  f"Resolved request: {json.dumps(request.__dict__)}\n"
                  f"Current asset: {json.dumps(current_asset) if current_asset else 'null'}\n"
                  f"Question: {question}\n\nExcerpts:\n{numbered_excerpts(citations)}")
        messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-16:]]
        messages.append({"role": "user", "content": prompt})
        h: HarnessResult = run_harness(
            self.llm, DIALOGUE_SYSTEM, messages,
            parse=lambda raw: self._parse(raw, request, len(citations)),
            check=lambda r: self._check(r, request),
            fallback=lambda: self._fallback(request, citations),
            max_attempts=self.max_attempts, log=log)
        result = h.result
        result.update(action=request.action, outcome=h.outcome, model_calls=len(h.attempts),
                      citations_used=[citations[i] for i in result.get("citation_refs", [])])
        return result

    @staticmethod
    def _parse(raw: str, request: Request, n: int) -> dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"_unparseable": True}
        refs = valid_refs(data.get("citation_refs"), n)
        asset = data.get("asset")
        if not (isinstance(asset, dict) and asset.get("type") in ASSET_FOR_FORMAT.values()
                and asset.get("provenance") == "generated_illustration" and asset.get("content")):
            asset = None
        m = data.get("misconception")
        try:
            confidence = float((m or {}).get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        keep = (isinstance(m, dict) and m.get("label") in MISCONCEPTION_LABELS and m.get("student_claim")
                and m.get("correction") and refs and confidence >= MIN_MISCONCEPTION_CONFIDENCE
                and request.action not in {"repair", "meta"})
        return {"answer": str(data.get("answer", "")), "citation_refs": refs, "asset": asset,
                "misconception": m if keep else None,
                "response_type": "generated_illustration" if asset else "grounded_text"}

    @staticmethod
    def _check(r: dict, request: Request) -> str | None:
        if r.get("_unparseable"):
            return "The response was not valid JSON."
        wanted = ASSET_FOR_FORMAT.get(request.output_format)
        if wanted and (r["asset"] is None or r["asset"]["type"] != wanted):
            return f"The request asks for a {wanted}. Return one valid {wanted} asset."
        if request.action in NEEDS_CITATION and not r["citation_refs"]:
            return "Use the supplied excerpts for factual claims and include the citation_refs."
        if not r["answer"].strip():
            return "The answer is empty."
        return None

    @staticmethod
    def _fallback(request: Request, citations: list[Citation]) -> dict:
        source = citations[0].text if citations else ""
        return {"answer": f"Let us restart with one small piece. The source says: {source} "
                          "Which part is unclear: the term, the rule, or an example?",
                "citation_refs": [0] if citations else [], "asset": None, "misconception": None,
                "response_type": "repair_fallback"}

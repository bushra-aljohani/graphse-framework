"""End-to-end walk-through of the framework on the synthetic corpus, fully offline.

Run from the repository root:  python examples/demo.py
Model output is scripted (ScriptedLLM); replace it with a real LLMClient to use a hosted model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphse.agents import DialogueTutor, InitialTutor, MCQAuthor, ScriptedLLM  # noqa: E402
from graphse.assessment import AssessmentService, ItemBank  # noqa: E402
from graphse.knowledge import build_graph, load_corpus, load_proposals  # noqa: E402
from graphse.orchestration import LearningController  # noqa: E402
from graphse.retrieval import BM25, GraphRetriever, HashingEmbedder, rrf  # noqa: E402
from graphse.store import LearnerStore  # noqa: E402


def main() -> None:
    # 1. Offline knowledge construction
    corpus = load_corpus(ROOT / "examples/toy_corpus.json")
    kg = build_graph(corpus, load_proposals(ROOT / "examples/toy_proposals.json"))
    print("Graph:", kg.counts())
    print("Rejected by the provenance gate:", [r.reason for r in kg.rejected])
    print("Relations not loaded (high-impact or inferred types):", kg.excluded_relations)

    # 2. Retrieval: graph retriever, BM25 and their fusion
    retriever = GraphRetriever(kg, HashingEmbedder())
    query = "how strongly do modules depend on each other"
    graph_rank = retriever.retrieve(query, min_score=0.1).ranking
    docs = {cid: c.name + " " + " ".join(p.span for p in c.provenance) for cid, c in kg.concepts.items()}
    lexical = BM25(docs).rank(query)
    embedding = retriever.similarities(query)
    name = lambda cid: kg.concepts[cid].name
    print("\nQuery:", query)
    print("  graph retriever:", [name(c) for c, _ in graph_rank[:3]])
    print("  BM25           :", [name(c) for c, _ in lexical[:3]])
    print("  RRF(BM25, emb) :", [name(c) for c, _ in rrf([lexical, embedding])[:3]])

    # 3. Online learning session (scripted model output)
    coupling = kg.concept_by_name("Coupling").concept_id
    llm = ScriptedLLM([
        {"explanation": "Coupling is how strongly modules depend on each other.",
         "example": "Two modules sharing one global variable are tightly coupled.", "citation_refs": [0]},
        {"answer": "Lower coupling means a change in one module rarely forces changes elsewhere.",
         "citation_refs": [0, 1], "asset": None, "misconception": None},
        {"question": "What does coupling measure?",
         "options": ["How strongly modules depend on each other", "How fast a module runs",
                     "How many classes a module has"],
         "correct_index": 0, "explanation": "Coupling measures dependence between modules.", "support_ref": 0},
    ])
    store = LearnerStore()
    controller = LearningController(kg, store, InitialTutor(llm), DialogueTutor(llm), MCQAuthor(llm),
                                    ItemBank(), AssessmentService(store, seed=1))
    sid = controller.start_session("learner-1")
    lesson = controller.explain("learner-1", sid, coupling)
    print("\nLesson grounded:", lesson["grounded"], "| cites:", [c.page for c in lesson["citations_used"]])
    turn = controller.chat("learner-1", sid, coupling, "Why does low coupling matter?",
                           client_citations=[{"text": "ignored: evidence comes from the server"}])
    print("Dialogue:", turn["outcome"], "| level:", turn["learner_level"], "| mastery records:", len(store.mastery_log))
    controller.finish_session("learner-1", sid)
    [question] = controller.start_test("learner-1", sid)
    print("Question:", question["question"], "| key visible to client:", "correct_index" in json.dumps(question))
    chosen = next(o["id"] for o in question["options"] if o["text"].startswith("How strongly"))
    result = controller.assessment.answer("learner-1", question["attempt_id"], chosen)
    print(f"Graded: {result['label']} | mastery {result['p_before']:.2f} -> {result['p_after']:.4f}")
    print("Score:", controller.complete_test("learner-1", sid))
    print("Recommendations:", [(name(r.concept_id), r.kind) for r in controller.recommendations("learner-1")])


if __name__ == "__main__":
    main()

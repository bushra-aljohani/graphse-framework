from graphse.retrieval import BM25, GraphRetriever, HashingEmbedder, citations_for_concept, rrf


def test_provenance_lookup_returns_only_the_concepts_own_spans(kg):
    cid = kg.concept_by_name("Coupling").concept_id
    cits = citations_for_concept(kg, cid)
    assert len(cits) == 2 and all(c.concept_id == cid for c in cits)
    assert citations_for_concept(kg, "unknown") == []


def test_rrf_formula():
    a = [("x", 9.0), ("y", 8.0)]
    b = [("y", 1.0), ("z", 0.5)]
    fused = dict(rrf([a, b], depth=20, k=10))
    assert abs(fused["y"] - (1 / 12 + 1 / 11)) < 1e-12
    assert abs(fused["x"] - 1 / 11) < 1e-12
    assert rrf([a, b])[0][0] == "y"


def test_rrf_depth_limits_contributions():
    assert "y" not in dict(rrf([[("x", 1.0), ("y", 0.5)]], depth=1))


def test_graph_retriever_is_bounded_and_traced(kg):
    r = GraphRetriever(kg, HashingEmbedder()).retrieve("modules depend on each other", min_score=0.1, max_nodes=3)
    assert r.stages["touched"] <= 3 and r.ranking
    assert all(c.concept_id in dict(r.ranking) for c in r.citations)


def test_bm25_prefers_lexical_match(kg):
    docs = {cid: c.name + " " + " ".join(p.span for p in c.provenance) for cid, c in kg.concepts.items()}
    top = BM25(docs).rank("boundary value analysis edges")[0][0]
    assert kg.concepts[top].name == "Boundary value analysis"

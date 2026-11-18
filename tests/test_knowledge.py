from dataclasses import replace

from graphse.knowledge import Proposal, check_proposal


def valid_node():
    return Proposal(kind="node", name="Coupling", node_type="Concept", chunk_id="A1.1-c1", page=10,
                    span="Coupling measures how strongly modules depend on each other.",
                    support_status="EXPLICIT", confidence=0.9)


def test_valid_node_passes(corpus):
    assert check_proposal(valid_node(), corpus.chunks).passed


def test_each_structural_defect_is_rejected(corpus):
    defects = {
        "span_not_verbatim": replace(valid_node(), span="Coupling is about speed."),
        "page_outside_chunk": replace(valid_node(), page=99),
        "unknown_chunk": replace(valid_node(), chunk_id="missing"),
        "unsupported_node_type": replace(valid_node(), node_type="Gadget"),
        "invalid_support_status": replace(valid_node(), support_status="MAYBE"),
        "empty_name": replace(valid_node(), name="  "),
        "confidence_out_of_range": replace(valid_node(), confidence=1.2),
    }
    for reason, proposal in defects.items():
        result = check_proposal(proposal, corpus.chunks)
        assert not result.passed and result.reason == reason


def test_relation_needs_accepted_endpoints(corpus):
    rel = Proposal(kind="relation", source="Coupling", target="Cohesion", relation_type="related_to",
                   chunk_id="A1.2-c1", page=11, span="Low coupling and high cohesion",
                   support_status="EXPLICIT", confidence=0.8)
    assert check_proposal(rel, corpus.chunks, {"coupling", "cohesion"}).passed
    assert check_proposal(rel, corpus.chunks, {"coupling"}).reason == "target_endpoint_not_accepted"


def test_graph_build_counts_and_relation_filter(kg):
    counts = kg.counts()
    assert counts["concepts"] == 10 and counts["rejected"] == 5
    assert counts["RELATED_TO"] == 2 and counts["IMPROVES"] == 1
    # high-impact relation types never enter the graph
    assert counts["excluded_relations"] == 1 and kg.review_queue == []
    assert not any(e.edge_type == "PREREQUISITE_OF" for e in kg.edges)


def test_optional_review_queue(corpus):
    # optional feature, not used in the paper: keep excluded relations for human review
    from graphse.knowledge import build_graph, load_proposals
    from conftest import ROOT
    kg = build_graph(corpus, load_proposals(ROOT / "examples/toy_proposals.json"), keep_review_queue=True)
    assert [p.relation_type for p in kg.review_queue] == ["prerequisite_of"]
    assert not any(e.edge_type == "PREREQUISITE_OF" for e in kg.edges)


def test_canonicalisation_keeps_all_provenance(kg):
    coupling = kg.concept_by_name("coupling")
    assert {p.section_id for p in coupling.provenance} == {"A1.1", "A1.2"}


def test_structural_edges_assert_location_only(kg):
    structural = [e for e in kg.edges if e.derivation == "structural"]
    assert structural and all(e.evidence is None for e in structural)
    assert all(e.edge_type in {"CO_LOCATED", "SEQUENCE_PRECEDES"} for e in structural)
    ep = kg.concept_by_name("Equivalence partitioning").concept_id
    bva = kg.concept_by_name("Boundary value analysis").concept_id
    assert ep in kg.predecessors(bva)          # adjacent sibling subtopics B1.1.1 -> B1.1.2

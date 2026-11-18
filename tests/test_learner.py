from graphse.learner import BKTParams, is_mastered, learner_level, recommend, update


def test_bkt_values_computed_by_hand():
    # correct: 0.27 / 0.41 = 0.658537, then + (1 - 0.658537) * 0.3
    assert abs(update(0.30, True) - 0.7609756097560976) < 1e-12
    # incorrect: 0.03 / 0.59 = 0.050847, then + (1 - 0.050847) * 0.3
    assert abs(update(0.30, False) - 0.33559322033898303) < 1e-12


def test_mastery_boundary():
    assert not is_mastered(0.9499) and is_mastered(0.95) and is_mastered(0.9501)
    assert is_mastered(0.80, BKTParams(mastery_threshold=0.8))


def test_levels():
    assert learner_level(0, {}) == "foundation"
    assert learner_level(2, {"a": 0.3}) == "developing"
    assert learner_level(5, {"a": 0.9, "b": 0.8}) == "proficient"


def test_recommendations_use_one_threshold():
    # a concept shown as not mastered (0.70 < 0.95) is still recommended
    recs = recommend({"c1": 0.70}, {}, {}, {}, {})
    assert [(r.concept_id, r.kind) for r in recs] == [("c1", "weakness")]
    assert recommend({"c1": 0.97}, {}, {}, {}, {}) == []


def test_recommendation_kinds():
    recs = recommend(mastery={"t": 0.5, "p": 0.2, "s": 0.85}, wrong={"t": 1}, open_misconceptions={"m": 2},
                     days_since_evidence={"s": 10}, predecessors={"t": ["p"]})
    kinds = {r.concept_id: r.kind for r in recs}
    assert kinds["p"] == "learning_order" and kinds["m"] == "misconception" and kinds["s"] == "slipping"

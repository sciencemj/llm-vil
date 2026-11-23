from llm_vil.agent.relationships import APPRAISAL_CLAMP, RelationshipStore


def store(owner="mina"):
    others = {"mina": "민아", "junho": "준호", "doyun": "도윤"}
    return RelationshipStore(owner, {k: v for k, v in others.items() if k != owner})


def test_relationships_are_directed():
    a, b = store("mina"), store("junho")
    a.apply_event("junho", "gift_received")
    assert a.get("junho").affection > 0
    assert b.get("mina").affection == 0


def test_event_table_and_bounds():
    s = store()
    for _ in range(10):
        s.apply_event("doyun", "witnessed_theft")
    r = s.get("doyun")
    assert -1.0 <= r.trust < -0.95 and r.affection >= -1.0  # diminishing returns, bounded
    assert len(r.recent_key_events) <= 5
    assert "믿기 어려움" in r.summary


def test_appraisal_clamped():
    s = store()
    s.appraise("junho", {"affection": 0.9, "trust": -0.7, "bogus": 3})
    r = s.get("junho")
    assert r.affection == APPRAISAL_CLAMP and r.trust == -APPRAISAL_CLAMP


def test_daily_decay_toward_neutral():
    s = store()
    s.apply_event("junho", "gift_received")
    s.apply_event("junho", "fair_trade")
    before = s.get("junho")
    s.daily_decay()
    after = s.get("junho")
    assert 0 < after.affection < before.affection
    assert 0 < after.trust < before.trust
    assert after.familiarity <= before.familiarity


def test_routine_positive_events_do_not_saturate():
    s = store()
    for _ in range(200):
        s.apply_event("junho", "pleasant_chat")
    r = s.get("junho")
    assert r.familiarity < 1.0 and r.affection < 1.0
    s.apply_event("junho", "witnessed_theft")  # opposite-direction push is not dampened
    assert s.get("junho").trust == -0.35

from llm_vil.agent.memory import HEARSAY_DECAY, MemoryStream
from llm_vil.llm.protocol import HashingEmbedder
from llm_vil.schemas.config import MemoryConfig

DAY = 1440


def stream(**kw) -> MemoryStream:
    return MemoryStream("mina", HashingEmbedder(), MemoryConfig(**kw))


def test_change_only_perception_cooldown():
    m = stream(perception_cooldown=60)
    assert m.perceive(0, "actor:junho", "준호가 보인다", 2) is not None
    assert m.perceive(30, "actor:junho", "준호가 보인다", 2) is None
    assert m.perceive(30, "actor:junho", "준호가 도둑질!", 8) is not None  # salient bypass
    assert m.perceive(100, "actor:junho", "준호가 보인다", 2) is not None


def test_retrieval_uses_subject_boost_and_relevance():
    m = stream()
    m.add(0, "observation", "밭에 물을 주었다", 2, subjects=["mina"])
    leo = m.add(10, "observation", "준호와 장작 이야기를 했다", 3, subjects=["junho"])
    m.add(20, "observation", "산딸기를 땄다", 2, subjects=["mina"])
    top = m.retrieve(30, "장작", subjects=["junho"], k=1)
    assert top == [leo]


def test_hearsay_confidence_decays_per_hop():
    a, b = stream(), MemoryStream("junho", HashingEmbedder(), MemoryConfig())
    orig = a.add(0, "observation", "도윤이 상자에서 산딸기를 가져갔다", 8, valence=-0.6,
                 subjects=["doyun", "haneul"])
    hop1 = b.add_hearsay(5, orig, "mina")
    assert hop1.kind == "hearsay" and hop1.source_actor_id == "mina"
    assert hop1.confidence == HEARSAY_DECAY
    c = MemoryStream("seoyeon", HashingEmbedder(), MemoryConfig())
    hop2 = c.add_hearsay(9, hop1, "junho")
    assert abs(hop2.confidence - HEARSAY_DECAY**2) < 1e-6
    assert orig.id in hop1.evidence_ids


def test_archive_moves_cold_records_and_keeps_permanent():
    m = stream(hot_days=3)
    old = m.add(0, "observation", "평범한 하루", 2, subjects=["mina"])
    keep = m.add(0, "observation", "도둑을 봤다", 8, subjects=["doyun"])
    refl = m.add(0, "reflection", "성실함이 중요하다", 6)
    archived = m.archive_old(4 * DAY)
    assert archived == [old]
    ids = {r.id for r in m.records}
    assert keep.id in ids and refl.id in ids and old.id not in ids
    assert any(r.kind == "summary" and old.id in r.evidence_ids for r in m.records)
    assert m.drain_pending() == [old] and m.drain_pending() == []


def test_reflection_threshold_and_snapshot():
    m = stream(reflection_threshold=20)
    for i in range(3):
        m.add(i, "observation", f"사건 {i}", 7)
    assert m.needs_reflection()
    m2 = stream(reflection_threshold=20)
    m2.restore(m.snapshot())
    assert m2.snapshot() == m.snapshot()

from llm_vil.engine.save_store import SCHEMA_VERSION, SaveStore
from llm_vil.schemas.agent import MemoryRecord
from llm_vil.schemas.world import WorldEvent


def rec(i, text, subjects, tick=0):
    return MemoryRecord(id=f"mina-m{i:05d}", owner_id="mina", tick=tick, kind="observation",
                        text=text, importance=3, subjects=subjects, embedding=[0.1, 0.2])


def test_migrations_wal_and_reopen(tmp_path):
    path = tmp_path / "save.sqlite"
    s = SaveStore(path)
    assert s.schema_version() == SCHEMA_VERSION
    assert s.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    s.close()
    s2 = SaveStore(path)  # idempotent migrations
    assert s2.schema_version() == SCHEMA_VERSION
    s2.close()


def test_archive_two_stage_search_korean(tmp_path):
    s = SaveStore(tmp_path / "a.sqlite")
    s.write_batch(archive=[
        (rec(1, "준호와 장작을 나눴다", ["junho"]), 100),
        (rec(2, "서연의 스튜가 맛있었다", ["seoyeon"]), 100),
        (rec(3, "연못가에서 산딸기를 땄다", ["mina"]), 100),
    ])
    by_subject = s.search_archive("mina", subject="junho", limit=3)
    assert [r.id for r in by_subject][0] == "mina-m00001"
    fts = s.search_archive("mina", query="스튜", limit=1)
    assert [r.id for r in fts] == ["mina-m00002"]
    assert fts[0].embedding == [0.1, 0.2]  # vectors survive archiving for cosine relevance
    raw = s.conn.execute("SELECT data FROM memory_archive LIMIT 1").fetchone()[0]
    assert "\\u" not in raw  # ensure_ascii=False
    # duplicates (re-simulated after restore) are ignored
    s.write_batch(archive=[(rec(1, "준호와 장작을 나눴다", ["junho"]), 100)])
    assert s.count("memory_archive") == 3


def test_checkpoints_and_truncate(tmp_path):
    s = SaveStore(tmp_path / "c.sqlite")
    s.save_checkpoint(10, '{"a": 1}')
    s.save_checkpoint(20, '{"a": 2}')
    s.write_batch(events=[WorldEvent(tick=t, kind="x") for t in (5, 15, 25)],
                  archive=[(rec(1, "late", ["mina"]), 25)])
    assert s.load_checkpoint() == (20, {"a": 2})
    assert s.load_checkpoint(15) == (10, {"a": 1})
    s.truncate_after(10)
    assert s.count("events") == 1 and s.count("checkpoints") == 1
    assert s.count("memory_archive") == 0
    assert s.search_archive("mina", query="late") == []

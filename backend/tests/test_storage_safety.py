"""Storage safety: bounded saves, compact snapshots, paused start and the run-day limit."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient

from llm_vil.cli import build_parser
from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import HashingEmbedder, TapeEntry, dumps
from llm_vil.run_options import run_options
from llm_vil.schemas.config import StorageConfig
from llm_vil.schemas.world import WorldEvent
from llm_vil.server.app import create_app
from llm_vil.server.runtime import MANUAL_LABEL, SimRuntime


def tape(n: int) -> list[TapeEntry]:
    return [TapeEntry(tick=i, agent_id="mina", purpose="probe", model="m", request="q" * 50,
                      response="{}", latency_ms=1.0, tokens_in=1, tokens_out=1, ok=True,
                      attempt=0) for i in range(n)]


def labels(store: SaveStore) -> list[str]:
    return [r["label"] for r in store.list_checkpoints(100)]


# ---------------------------------------------------------------- SQLite bounds
def test_auto_checkpoints_are_pruned_and_labeled_saves_survive(tmp_path):
    s = SaveStore(tmp_path / "s.sqlite")
    for i in range(25):
        s.save_checkpoint(i * 10, "{}", "중요" if i in (0, 3) else "")
    s.maintain(StorageConfig(max_auto_checkpoints=4))
    rows = s.list_checkpoints(100)
    assert [r["tick"] for r in rows if r["label"] == ""] == [240, 230, 220, 210]
    assert [r["tick"] for r in rows if r["label"]] == [30, 0]
    s.maintain(StorageConfig(max_auto_checkpoints=4))  # idempotent
    assert s.count("checkpoints") == 6
    s.close()


async def test_engine_prunes_after_each_automatic_checkpoint(config, tmp_path):
    cfg = config.model_copy(update={"storage": StorageConfig(max_auto_checkpoints=2)})
    s = SaveStore(tmp_path / "s.sqlite")
    e = SimEngine(cfg, DemoBackend(), store=s, checkpoint_every_days=1)
    e.checkpoint("내 저장")
    await e.run_days(5)
    assert labels(s).count("") == 2 and "내 저장" in labels(s)
    s.close()


async def test_unnamed_manual_save_is_labeled_so_it_is_never_pruned(config, tmp_path):
    s = SaveStore(tmp_path / "s.sqlite")
    rt = SimRuntime(SimEngine(config, DemoBackend(), store=s))
    meta = await rt.save_checkpoint("")
    assert meta["label"] == MANUAL_LABEL
    assert (await rt.command({"cmd": "checkpoint"}))["ok"]
    for i in range(15):
        s.save_checkpoint(5000 + i, "{}")
    s.maintain(StorageConfig())
    assert labels(s).count(MANUAL_LABEL) == 2 and labels(s).count("") == 10
    s.close()


def test_tape_events_and_embed_cache_are_capped(tmp_path):
    s = SaveStore(tmp_path / "s.sqlite")
    s.write_batch(tape=tape(500),
                  events=[WorldEvent(tick=i, kind="chopped") for i in range(500)])
    for i in range(60):
        s.put_vector(f"k{i}", "m", 2, [0.0, 1.0])
    limits = StorageConfig(max_tape_rows=100, max_events_rows=50, max_embed_cache_rows=20)
    s.maintain(limits)
    assert s.count("tape") == 100 and s.count("embed_cache") == 20
    assert s.count("events") == 500  # no checkpoint yet: nothing may be dropped
    assert s.conn.execute("SELECT MIN(tick) FROM tape").fetchone()[0] == 400  # newest kept
    s.save_checkpoint(300, "{}")
    s.maintain(limits)
    # everything after the newest checkpoint stays although it is over the cap
    assert s.conn.execute("SELECT MIN(tick), COUNT(*) FROM events").fetchone() == (301, 199)
    s.save_checkpoint(499, "{}")
    s.maintain(limits)
    assert s.count("events") == 50
    s.maintain(StorageConfig(max_tape_rows=0))
    assert s.count("tape") == 0
    s.close()


def test_incremental_vacuum_shrinks_the_file(tmp_path):
    path = tmp_path / "s.sqlite"
    s = SaveStore(path)
    assert s._pragma("auto_vacuum") == 2
    for i in range(30):
        s.save_checkpoint(i, "x" * 500_000)
    s.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = path.stat().st_size
    s.maintain(StorageConfig(max_auto_checkpoints=3))
    assert s._pragma("freelist_count") == 0
    s.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert path.stat().st_size < before / 5
    assert s.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    s.reset_all()
    s.close()


def test_old_save_without_auto_vacuum_is_converted_or_left_usable(tmp_path, monkeypatch):
    def legacy(path):
        c = sqlite3.connect(path)
        c.execute("PRAGMA auto_vacuum = NONE")
        c.execute("CREATE TABLE legacy (x)")
        c.execute("INSERT INTO legacy VALUES (1)")
        c.commit()
        c.close()

    legacy(tmp_path / "small.sqlite")
    s = SaveStore(tmp_path / "small.sqlite")
    assert s._pragma("auto_vacuum") == 2
    assert s.conn.execute("SELECT x FROM legacy").fetchone() == (1,)
    s.close()

    monkeypatch.setattr("llm_vil.engine.save_store.CONVERT_MAX_BYTES", 0)  # "too large"
    legacy(tmp_path / "big.sqlite")
    s = SaveStore(tmp_path / "big.sqlite")
    assert s._pragma("auto_vacuum") == 0
    s.save_checkpoint(1, "{}")
    s.maintain(StorageConfig())  # no error, nothing lost
    assert s.count("checkpoints") == 1
    s.close()


# ---------------------------------------------------------------- snapshot size
async def test_snapshot_stays_bounded_for_30_days_and_restores_identically(config):
    e = SimEngine(config, DemoBackend())
    sizes = {}
    for day in range(1, 31):
        await e.run_days(1)
        sizes[day] = len(dumps(e.snapshot()).encode())
    assert max(sizes.values()) < 1_500_000, sizes
    assert sizes[30] < sizes[10] * 1.3, sizes  # plateau (was +83 KB every day)
    m = config.memory
    for a in e.agents.values():
        for kind in ("reflection", "summary", "fact", "goal"):
            # caps are enforced daily: allow what one day adds on top
            assert len(a.memory.of_kind(kind)) <= m.permanent_cap(kind) + 30
    # compact snapshot still restores to the same minds (vectors included) and future
    data = e.snapshot()
    assert '"embedding"' not in dumps(data)
    b = SimEngine(config, DemoBackend())
    b.load_snapshot(data)
    assert dumps(b.snapshot()) == dumps(data)
    for aid in e.order:
        assert b.agents[aid].memory.records == e.agents[aid].memory.records
    await e.run_days(1)
    await b.run_days(1)
    assert dumps(e.snapshot()) == dumps(b.snapshot())


async def test_evicted_memories_stay_searchable_in_the_archive(config, tmp_path):
    mem = config.memory.model_copy(update={"max_reflections": 3})
    cfg = config.model_copy(update={"memory": mem})
    s = SaveStore(tmp_path / "s.sqlite")
    e = SimEngine(cfg, DemoBackend(), store=s)
    for i in range(6):
        e.agents["mina"].memory.add(e.tick, "reflection", f"깨달음 번호{i}", 5)
    e._daily_maintenance()
    kept = [r.text for r in e.agents["mina"].memory.of_kind("reflection")]
    assert kept == ["깨달음 번호3", "깨달음 번호4", "깨달음 번호5"]
    assert s.search_archive("mina", query="번호0")[0].text == "깨달음 번호0"
    s.close()


async def test_blocking_embedder_recomputes_vectors_after_restore(config):
    class Blocking(HashingEmbedder):
        blocking = True

        async def aembed(self, texts):  # type: ignore[no-untyped-def]
            return [self.embed(t) for t in texts]

    a = SimEngine(config, DemoBackend(), embedder=Blocking())
    a.agents["mina"].memory.add(a.tick, "observation", "숲에서 나무를 봤다", 5)
    await a._embed_pending()
    b = SimEngine(config, DemoBackend(), embedder=Blocking())
    b.load_snapshot(a.snapshot())
    assert b.agents["mina"].memory.pending_embeddings()
    await b._embed_pending()
    assert b.agents["mina"].memory.records == a.agents["mina"].memory.records


# ---------------------------------------------------------------- paused by default
def test_server_starts_paused_and_holds_the_clock_until_resume(config):
    app = create_app(config, DemoBackend(), speed=8.0)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 5000)) as c:
        import time

        t0 = c.get("/api/health").json()
        assert t0["paused"] and t0["max_run_days"] == 7 and not t0["limit_reached"]
        time.sleep(0.5)
        assert c.get("/api/health").json()["tick"] == t0["tick"]
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"cmd": "resume"})
            deadline = time.monotonic() + 10
            while c.get("/api/health").json()["tick"] == t0["tick"]:
                assert time.monotonic() < deadline
                time.sleep(0.05)
            ws.send_json({"cmd": "pause"})


@pytest.mark.parametrize(("argv", "paused"), [
    ([], True), (["--paused"], True), (["--no-paused"], False), (["--autostart"], False),
])
def test_cli_defaults_to_paused(argv, paused):
    assert run_options(build_parser().parse_args(["serve", *argv]))["paused"] is paused


def test_cli_limits(config):
    opts = run_options(build_parser().parse_args(
        ["serve", "--max-days", "0", "--max-checkpoints", "3"]))
    assert opts["max_run_days"] == 0 and opts["max_checkpoints"] == 3
    assert "max_run_days" not in run_options(build_parser().parse_args(["serve"]))
    for bad in (["--max-days", "-1"], ["--max-checkpoints", "0"]):
        with pytest.raises(SystemExit):
            run_options(build_parser().parse_args(["serve", *bad]))
    app = create_app(config, DemoBackend(), max_run_days=0, max_checkpoints=3)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        rt = app.state.runtime
        assert rt.max_run_days is None and rt.engine.storage.max_auto_checkpoints == 3
        assert c.get("/api/snapshot").json()["clock"]["max_run_days"] is None


# ---------------------------------------------------------------- run-day limit
async def run_unattended(rt: SimRuntime) -> None:
    """Let the real background loop run flat out until it pauses itself."""
    rt.pacer.set_speed(None)
    rt.start()
    for _ in range(6000):
        await asyncio.sleep(0.01)
        if rt.paused:
            return
    raise AssertionError("never paused")


async def test_max_run_days_pauses_exactly_at_the_limit_and_resume_grants_more(config):
    e = SimEngine(config, DemoBackend())
    rt = SimRuntime(e, max_run_days=1)
    start = e.tick
    assert (await rt.command({"cmd": "resume"}))["ok"]
    await run_unattended(rt)
    assert e.tick == start + 1440 and rt.limit_reached
    h = rt.health()
    assert h["limit_reached"] and h["run_days_elapsed"] == 1.0 and h["paused"]
    notices = [ev for ev in rt.timeline if ev["kind"] == "run_limit_reached"]
    assert len(notices) == 1
    assert notices[0]["data"]["text"] == "최대 실행 일수(1일)에 도달해 자동 일시정지되었습니다"
    await asyncio.sleep(0.1)
    assert e.tick == start + 1440  # really stopped
    assert not (await rt.command({"cmd": "step", "ticks": 10}))["ok"]

    assert (await rt.command({"cmd": "resume"}))["ok"]  # fresh window of one more day
    assert not rt.limit_reached
    await run_unattended(rt)
    assert e.tick == start + 2880 and rt.limit_reached
    await rt.stop()


async def test_set_max_days_command(config):
    e = SimEngine(config, DemoBackend())
    rt = SimRuntime(e, max_run_days=1)
    for _ in range(1440):
        await rt.tick_once()
    assert rt.limit_reached and rt.paused
    assert (await rt.command({"cmd": "set_max_days", "days": 3}))["ok"]
    assert not rt.limit_reached and rt.paused and rt.health()["max_run_days"] == 3
    rt.paused = False
    assert (await rt.command({"cmd": "set_max_days", "days": 1}))["ok"]  # already used up
    assert rt.limit_reached and rt.paused
    for unlimited in (0, None):
        assert (await rt.command({"cmd": "set_max_days", "days": unlimited}))["ok"]
        assert rt.max_run_days is None and not rt.limit_reached
    for bad in (-1, 1.5, "x", True, 10**9, [1], float("inf"), float("nan")):
        assert not (await rt.command({"cmd": "set_max_days", "days": bad}))["ok"]


async def test_rewind_restarts_the_run_window(config, tmp_path):
    s = SaveStore(tmp_path / "s.sqlite")
    e = SimEngine(config, DemoBackend(), store=s)
    rt = SimRuntime(e, max_run_days=1)
    cid = (await rt.save_checkpoint("시작"))["id"]
    for _ in range(1440):
        await rt.tick_once()
    assert rt.limit_reached
    await rt.restore_checkpoint(checkpoint_id=cid)
    assert not rt.limit_reached and rt.health()["run_days_elapsed"] == 0
    s.close()

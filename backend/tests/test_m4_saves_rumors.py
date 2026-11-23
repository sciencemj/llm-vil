"""M4: checkpoint management API, rewind/reset broadcast, organic rumor propagation,
merchant coin sink."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_vil.agent.conversation import ConversationSession
from llm_vil.agent.memory import HEARSAY_DECAY, rumor_root
from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.schemas.agent import ConversationTurn
from llm_vil.server import create_app

DAY = 1440


@pytest.fixture
def live(config, tmp_path):
    app = create_app(config, paused=True, store_path=str(tmp_path / "village.sqlite"),
                     t_max_seconds=0.0, speed=8.0)
    with TestClient(app) as c:
        yield c, app


def step(ws, ticks: int) -> int:
    """Advance a paused village ``ticks`` sim minutes; returns the reached tick."""
    ws.send_text(json.dumps({"cmd": "step", "ticks": ticks}))
    start = None
    while True:
        m = ws.receive_json()
        if m["type"] != "batch":
            continue
        clocks = [d["tick"] for d in m["deltas"] if d["type"] == "clock"]
        if not clocks:
            continue
        if start is None:
            start = min(clocks) - 1
        if max(clocks) >= start + ticks:
            return max(clocks)


# ---------------------------------------------------------------- save store
def test_checkpoint_labels_and_listing(tmp_path):
    s = SaveStore(tmp_path / "s.sqlite")
    a = s.save_checkpoint(400, "{}", "아침 산책 전")
    b = s.save_checkpoint(1500, "{}")
    rows = s.list_checkpoints()
    assert [r["id"] for r in rows] == [b, a]
    assert rows[1] == {**rows[1], "tick": 400, "day": 0, "minute": 400, "label": "아침 산책 전"}
    assert rows[0]["day"] == 1 and rows[0]["label"] == "" and rows[0]["created_at"]
    s.truncate_after(500, keep_checkpoints=True)
    assert s.count("checkpoints") == 2


# ---------------------------------------------------------------- API
def test_save_advance_restore_broadcasts_snapshot(live):
    c, app = live
    info = c.get("/api/saves/checkpoints").json()
    assert info["persistent"] and info["save_path"].endswith("village.sqlite")
    start = info["current_tick"]
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        step(ws, 30)
        made = c.post("/api/saves/checkpoint", json={"label": "첫 저장"}).json()
        saved = made["created"]
        assert saved["label"] == "첫 저장" and saved["tick"] >= start + 30
        assert made["checkpoints"][0]["id"] == saved["id"]
        step(ws, 90)
        assert c.get("/api/health").json()["tick"] > saved["tick"] + 60
        r = c.post("/api/saves/restore", json={"id": saved["id"]})
        assert r.status_code == 200 and r.json()["restored"]["tick"] == saved["tick"]
        while True:  # every connected client gets a fresh full snapshot
            m = ws.receive_json()
            if m["type"] == "snapshot":
                assert m["clock"]["tick"] == saved["tick"]
                break
    assert c.get("/api/health").json()["tick"] == saved["tick"]
    # restoring by tick picks the latest save at/before it
    assert c.post("/api/saves/restore", json={"tick": saved["tick"]}).status_code == 200
    assert len(c.get("/api/saves/checkpoints").json()["checkpoints"]) == 1
    assert c.post("/api/saves/restore", json={"id": 999}).status_code == 400
    assert c.post("/api/saves/restore", json={}).status_code == 400
    assert app.state.runtime.engine.metrics["errors"] == 0


def test_reset_returns_to_day_one(live, config):
    c, _ = live
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        step(ws, 60)
        c.post("/api/saves/checkpoint", json={"label": "before reset"})
        r = c.post("/api/saves/reset").json()
        assert r["reset"]["tick"] == config.start_minute and r["reset"]["day"] == 0
        while ws.receive_json()["type"] != "snapshot":
            pass
    assert c.get("/api/saves/checkpoints").json()["checkpoints"] == []  # reset wipes saves
    assert c.get("/api/agent/mina").json()["recent_memories"] == []


def test_saves_api_is_local_only_and_memory_only_mode(live, config):
    c, _ = live
    for method, path in (("get", "/api/saves/checkpoints"), ("post", "/api/saves/checkpoint"),
                         ("post", "/api/saves/restore"), ("post", "/api/saves/reset")):
        r = getattr(c, method)(path, headers={"host": "evil.example"},
                               **({"json": {}} if method == "post" else {}))
        assert r.status_code == 403, path
        r = getattr(c, method)(path, headers={"origin": "https://evil.example"},
                               **({"json": {}} if method == "post" else {}))
        assert r.status_code == 403, path
    app = create_app(config, paused=True)  # no store
    with TestClient(app) as mc:
        info = mc.get("/api/saves/checkpoints").json()
        assert not info["persistent"] and info["checkpoints"] == []
        assert mc.post("/api/saves/checkpoint", json={"label": "x"}).status_code == 400


# ---------------------------------------------------------------- rumors
def _turn(engine: SimEngine, speaker: str, listener: str, shared) -> None:
    from llm_vil.agent.agent import TurnDecision

    sid = "conv_test"
    s = ConversationSession(id=sid, initiator_id=speaker, target_id=listener, topic="소문",
                            start_tick=engine.tick)
    s.accept(engine.tick)
    engine.sessions[sid] = s
    engine._apply_turn(sid, speaker, TurnDecision(turn=ConversationTurn(
        speaker_id=speaker, listener_id=listener, tick=engine.tick, utterance="그거 알아?"),
        shared=shared))
    engine.sessions.pop(sid, None)


def test_rumor_decays_per_hop_and_never_echoes(config):
    e = SimEngine(config, DemoBackend())
    e.inject_rumor("junho", "doyun", "도윤이 밤에 창고를 뒤졌대")
    god = e.agents["junho"].rumors()[-1]
    assert god.source_actor_id == "rumor" and god.confidence == pytest.approx(0.7)
    assert god in e.agents["junho"]._shareable("mina")
    assert god not in e.agents["junho"]._shareable("doyun")  # never gossip to its subject
    _turn(e, "junho", "mina", god)
    heard = e.agents["mina"].rumors()[-1]
    assert heard.source_actor_id == "junho" and rumor_root(heard) == god.id
    assert heard.confidence == pytest.approx(0.7 * HEARSAY_DECAY)
    assert "에게 들은 이야기" not in heard.text.split(": ", 1)[1]  # no nested prefixes
    assert god not in e.agents["junho"]._shareable("mina")  # already told
    _turn(e, "mina", "junho", heard)  # telling it back is not a new rumor
    assert len(e.agents["junho"].rumors()) == 1 and e.metrics["hearsay_shared"] == 1
    # third hop drops below the 0.35 threshold and stops spreading
    _turn(e, "mina", "seoyeon", heard)
    third = e.agents["seoyeon"].rumors()[-1]
    assert third.confidence == pytest.approx(0.343, abs=1e-3)
    assert third not in e.agents["seoyeon"]._shareable("haneul")


async def test_god_rumor_spreads_organically_through_demo_conversations(config):
    e = SimEngine(config, DemoBackend())
    await e.run_until(config.start_minute + 60)
    e.inject_rumor("seoyeon", "doyun", "도윤이 장터 돈을 슬쩍했대")
    root = e.agents["seoyeon"].rumors()[-1].id
    await e.run_days(3)
    carriers = [aid for aid, a in e.agents.items() if aid != "seoyeon"
                and any(rumor_root(r) == root for r in a.rumors())]
    assert carriers and e.metrics["hearsay_shared"] > 0
    for aid in carriers:
        r = next(r for r in e.agents[aid].rumors() if rumor_root(r) == root)
        assert r.confidence <= 0.7 * HEARSAY_DECAY + 1e-9 and r.source_actor_id in e.agents


def test_inspector_shows_rumors_and_goals(live):
    c, _ = live
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"cmd": "god_rumor", "target": "mina", "about": "junho",
                                 "text": "준호가 숲에서 뭔가 묻었대"}))
        while (m := ws.receive_json())["type"] != "ack":
            pass
        assert m["ok"]
    d = c.get("/api/agent/mina").json()
    assert d["rumors"][0]["source"] == "신의 속삭임"
    assert d["rumors"][0]["confidence"] == 0.7 and d["rumors"][0]["about"] == ["junho"]
    assert "goal_details" in d


# ---------------------------------------------------------------- economy
def test_demo_spends_surplus_coins_at_the_merchant():
    demo = DemoBackend()
    ctx = {
        "self": {"id": "mina", "bed_id": "bed_mina", "sleep_minute": 1260, "wake_minute": 330,
                 "generosity": 0.2, "sociability": 0.0, "work_bias": {"farm": 1.0},
                 "home_area": "home_mina", "impulsiveness": 0.0, "chest_id": "chest_mina"},
        "minute": 600, "day": 3, "reason": "idle",
        "options": {"trade": {"targets": ["stall_main"],
                              "items": ["bread", "turnip_seed", "wheat_seed"]},
                    "visit_and_talk": {"targets": []}, "wander": {"targets": ["plaza"]}},
        "inventory": {"coins": 80, "turnip_seed": 5}, "needs": {"hunger": 10, "stamina": 90},
        "item_prices": {"bread": {"buy": 6}, "turnip_seed": {"buy": 2}, "wheat_seed": {"buy": 3}},
        "thresholds": {"hungry": 50, "tired": 25}, "orders": [], "talked_today": [],
        "unanswered": [], "relationships": {}, "visible_actors": [], "plan_block": None,
        "known_chests": {},
    }
    choice = demo._choose_routine(ctx)
    assert choice["routine"] == "trade" and choice["op"] == "buy" and choice["item"] == "bread"
    ctx["inventory"]["coins"] = 20
    assert demo._choose_routine(ctx)["routine"] != "trade"


# ================================================================ M4 review regressions
def _archive_rows(store: SaveStore) -> list[tuple]:
    return store.conn.execute(
        "SELECT agent_id, memory_id, archived_tick FROM memory_archive ORDER BY rowid").fetchall()


def test_archive_survives_restart_and_restore_without_branch_mixing(config, tmp_path):
    path = str(tmp_path / "village.sqlite")
    app = create_app(config, paused=True, store_path=path, t_max_seconds=0.0)
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(2):
            step(ws, DAY)  # -> day 2 (the server caps one step at 1440 ticks)
        early = c.post("/api/saves/checkpoint", json={"label": "day2"}).json()["created"]
        for _ in range(3):
            step(ws, DAY)  # -> day 5: memories older than 3 days are archived
        late = c.post("/api/saves/checkpoint", json={"label": "day5"}).json()["created"]
    store = SaveStore(path)
    rows_at_late = _archive_rows(store)
    assert any(t >= 4 * DAY for _, _, t in rows_at_late)  # Day 4+ archive rows exist
    store.close()

    # restart: resumes the latest checkpoint with its whole archive intact
    app2 = create_app(config, paused=True, store_path=path, t_max_seconds=0.0)
    with TestClient(app2) as c2:
        assert c2.get("/api/health").json()["tick"] == late["tick"]
        store2 = SaveStore(path)  # own connection: the app's lives in the server thread
        assert _archive_rows(store2) == rows_at_late
        # restoring the latest save again loses nothing
        assert c2.post("/api/saves/restore", json={"id": late["id"]}).status_code == 200
        assert _archive_rows(store2) == rows_at_late
        # restoring the earlier save rewinds archive *and* later checkpoints together
        assert c2.post("/api/saves/restore", json={"id": early["id"]}).status_code == 200
        assert all(t <= early["tick"] for _, _, t in _archive_rows(store2))
        cps = c2.get("/api/saves/checkpoints").json()["checkpoints"]
        assert early["id"] in [cp["id"] for cp in cps] and late["id"] not in [
            cp["id"] for cp in cps]
        # earlier (auto) save points remain; nothing from the abandoned future does
        assert all(cp["tick"] <= early["tick"] for cp in cps)
        # reset: a clean day-1 village with no saves, archive or log left
        c2.post("/api/saves/reset")
        assert store2.count("checkpoints") == store2.count("memory_archive") == 0
        assert store2.count("events") == 0

    # --fresh wipes a save explicitly; default resume with no checkpoints starts on day 1
    app3 = create_app(config, paused=True, store_path=path, fresh=True)
    with TestClient(app3) as c3:
        assert c3.get("/api/health").json()["tick"] == config.start_minute


def test_two_hop_rumor_keeps_subjects_and_spares_the_messengers(config):
    e = SimEngine(config, DemoBackend())
    e.inject_rumor("junho", "doyun", "도윤이 밤에 창고를 뒤졌대")
    first = e.agents["junho"].rumors()[-1]
    trust_before = {a: e.agents[a].relationships.get("junho").trust for a in ("mina", "seoyeon")}
    _turn(e, "junho", "mina", first)
    hop1 = e.agents["mina"].rumors()[-1]
    assert hop1.subjects == ["doyun"] and hop1.source_actor_id == "junho"
    assert hop1.text.startswith("준호에게 들은 이야기: ")  # Korean display name, not "junho"
    assert hop1 not in e.agents["mina"]._shareable("junho")  # never back to the teller
    _turn(e, "mina", "seoyeon", hop1)
    hop2 = e.agents["seoyeon"].rumors()[-1]
    assert hop2.subjects == ["doyun"] and hop2.source_actor_id == "mina"
    assert hop2.text.startswith("민아에게 들은 이야기: ")
    for a in ("mina", "seoyeon"):
        assert e.agents[a].relationships.get("junho").trust == trust_before[a]
    assert e.agents["seoyeon"].relationships.get("mina").trust == 0.0
    assert e.agents["seoyeon"].relationships.get("doyun").trust < 0  # only the subject


def test_restore_requeues_vectors_from_another_embedder(config):
    from llm_vil.llm.embeddings import CachingEmbedder

    class Blocking(CachingEmbedder):
        provider = "local"
        blocking = True

        def __init__(self):
            super().__init__("other-model", 8)

        def _compute(self, texts):
            return [[1.0] * 8 for _ in texts]

    old = SimEngine(config, DemoBackend())  # hashing vectors (64-d)
    old.agents["mina"].memory.add(old.tick, "observation", "저장 전 기억", 3)
    snap = old.snapshot()
    new = SimEngine(config, DemoBackend(), embedder=Blocking())
    new.load_snapshot(snap)
    mem = new.agents["mina"].memory
    assert [rid for rid, _ in mem.pending_embeddings()] == [mem.records[-1].id]


def test_restore_does_not_persist_abandoned_events(live):
    c, app = live
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        step(ws, 10)
        saved = c.post("/api/saves/checkpoint", json={}).json()["created"]
        step(ws, 20)
    store = SaveStore(app.state.runtime.engine.store.path)  # test-thread connection
    c.post("/api/saves/restore", json={"id": saved["id"]})
    late = store.conn.execute("SELECT COUNT(*) FROM events WHERE tick > ?",
                              (saved["tick"],)).fetchone()[0]
    assert late == 0 and app.state.runtime.engine.pending_events == []

"""Replay recording: frame store, /api/replay/*, local-only guard, rewind/reset truncation."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.server import create_app
from llm_vil.server.replay import ReplayRecorder
from llm_vil.server.runtime import SimRuntime


def step(ws, ticks: int) -> int:
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


@pytest.fixture
def live(config, tmp_path):
    app = create_app(config, paused=True, store_path=str(tmp_path / "v.sqlite"),
                     t_max_seconds=0.0, speed=8.0)
    with TestClient(app) as c:
        yield c, app


async def test_runtime_records_a_frame_per_tick_and_store_batches(config, tmp_path):
    store = SaveStore(tmp_path / "r.sqlite")
    e = SimEngine(config, DemoBackend(), store=store)
    rt = SimRuntime(e, paused=True)
    start = e.tick
    for _ in range(45):
        await rt.tick_once()
    # store-backed: 30 frames flushed in one batch, 15 still pending
    assert store.count("replay_frames") == 30 == len(store.frame_ticks())
    ticks = rt.replay.ticks()
    assert ticks == list(range(start + 1, start + 46))  # pending frames are visible too
    frame = rt.replay_frame(start + 10)
    assert frame["type"] == "snapshot" and frame["replay"] and frame["clock"]["paused"]
    assert frame["clock"]["tick"] == start + 10
    assert {a["actor_id"] for a in frame["actors"]} == set(e.order)
    assert all({"x", "y", "facing", "activity", "routine", "hunger", "speech"} <= set(a)
               for a in frame["actors"])
    assert len(frame["objects"]) == len(e.world.objects)
    assert all({"id", "kind", "pos", "state"} <= set(o) for o in frame["objects"])
    rt.replay.flush()
    assert len(store.frame_ticks()) == 45


async def test_frames_are_capped_and_memory_only_mode_works(config, tmp_path):
    store = SaveStore(tmp_path / "c.sqlite")
    rec = ReplayRecorder(store, cap=50)
    rec._pending = [(t, 0, t, "{}") for t in range(1, 121)]
    rec.flush()
    assert store.frame_ticks() == list(range(71, 121))  # newest 50 kept
    e = SimEngine(config, DemoBackend())  # no store
    rt = SimRuntime(e, paused=True, replay_cap=20)
    for _ in range(30):
        await rt.tick_once()
    assert len(rt.replay.ticks()) == 20 and rt.replay_frame(e.tick)["clock"]["tick"] == e.tick


def test_replay_endpoints_exact_nearest_and_404(live):
    c, _ = live
    empty = c.get("/api/replay/range").json()
    assert empty["count"] == 0 and empty["min_tick"] is None
    assert c.get("/api/replay/frame", params={"tick": 10}).status_code == 404
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        end = step(ws, 40)
    assert "ticks" not in c.get("/api/replay/range").json()  # compact by default
    r = c.get("/api/replay/range", params={"include_ticks": 1}).json()
    assert r["count"] >= 40 and r["max_tick"] == end and r["ticks"][-1] == end
    assert r["min"]["day"] == 0 and r["live_tick"] == end
    mid = r["ticks"][len(r["ticks"]) // 2]
    exact = c.get("/api/replay/frame", params={"tick": mid}).json()
    assert exact["clock"]["tick"] == mid and exact["replay"] is True
    nearest = c.get("/api/replay/frame", params={"tick": end + 500}).json()
    assert nearest["clock"]["tick"] == end  # nearest frame at or before the request
    assert c.get("/api/replay/frame", params={"tick": r["min_tick"] - 1}).status_code == 404
    text = c.get("/api/replay/frame", params={"tick": mid}).text
    assert "\\u" not in text  # Korean stays readable (ensure_ascii=False)


def test_replay_is_local_only(live):
    c, _ = live
    for path, params in (("/api/replay/range", {}), ("/api/replay/frame", {"tick": 1})):
        assert c.get(path, params=params, headers={"host": "evil.example"}).status_code == 403
        assert c.get(path, params=params,
                     headers={"origin": "https://evil.example"}).status_code == 403


def test_restore_and_reset_truncate_frames(live, config):
    c, app = live
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        step(ws, 20)
        saved = c.post("/api/saves/checkpoint", json={"label": "r"}).json()["created"]
        step(ws, 40)
        assert c.get("/api/replay/range").json()["max_tick"] > saved["tick"] + 30
        c.post("/api/saves/restore", json={"id": saved["id"]})
        r = c.get("/api/replay/range").json()
        assert r["max_tick"] == saved["tick"]  # the abandoned future's frames are gone
        step(ws, 5)
        assert c.get("/api/replay/range").json()["max_tick"] == saved["tick"] + 5
        c.post("/api/saves/reset")
        assert c.get("/api/replay/range").json()["count"] == 0
    store = SaveStore(app.state.runtime.engine.store.path)
    assert store.frame_ticks() == []


def test_gemini_defaults_updated():
    import yaml

    from llm_vil.config import CONFIG_DIR

    # the shipped defaults only (a developer's gitignored llm.local.yaml may override them)
    g = yaml.safe_load((CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))["profiles"]["gemini"]
    assert g["fast_model"] == "gemini-3.5-flash-lite" and g["smart_model"] == "gemini-3.8-flash"


# ================================================================ review regressions
async def test_reset_does_not_leak_old_speech_or_events_into_new_frames(config):
    e = SimEngine(config, DemoBackend())
    rt = SimRuntime(e, paused=True)
    for _ in range(5):
        await rt.tick_once()
    # simulate a line of dialogue and a pending event right before the reset
    rt._speech["mina"] = (e.tick, "리셋 전에 한 말")
    rt._tick_events = [{"type": "event", "kind": "utterance", "tick": e.tick}]
    await rt.reset_village()
    await rt.tick_once()
    first = rt.replay_frame(e.tick)
    assert first["clock"]["tick"] == config.start_minute + 1
    assert all(a["speech"] is None for a in first["actors"])
    assert all(ev.get("tick", 0) > config.start_minute for ev in first["events"])
    assert "리셋 전에 한 말" not in str(first)


async def test_future_speech_never_shows_on_earlier_ticks(config):
    e = SimEngine(config, DemoBackend())
    rt = SimRuntime(e, paused=True)
    rt._speech["junho"] = (e.tick + 50, "미래의 말")  # e.g. left over from a later timeline
    await rt.tick_once()
    frame = rt.replay_frame(e.tick)
    assert next(a for a in frame["actors"] if a["actor_id"] == "junho")["speech"] is None


def test_checkpoint_flushes_buffered_frames(live):
    c, app = live
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        step(ws, 12)  # fewer than one write batch: frames are still buffered
    c.post("/api/saves/checkpoint", json={})
    store = SaveStore(app.state.runtime.engine.store.path)
    assert store.count("replay_frames") >= 12

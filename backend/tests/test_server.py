"""Live server: REST endpoints, WebSocket snapshot + delta stream, control & God commands."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_vil.server import create_app


@pytest.fixture
def client(config, tmp_path):
    app = create_app(config, speed=8.0, paused=True, store_path=str(tmp_path / "live.sqlite"),
                     t_max_seconds=0.0)
    with TestClient(app) as c:
        yield c


def collect(ws, until, limit: int = 400) -> list[dict[str, Any]]:
    """Receive messages until ``until(msg)`` matches; return everything seen."""
    seen = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if until(msg):
            return seen
    raise AssertionError(f"condition not met in {limit} messages")


def deltas(msgs):
    return [d for m in msgs if m["type"] == "batch" for d in m["deltas"]]


def test_health_snapshot_and_agent(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["actors_count"] == 5 and h["paused"] and h["speed"] == 8.0
    snap_resp = client.get("/api/snapshot")
    assert "\\u" not in snap_resp.text  # ensure_ascii=False on the wire
    snap = snap_resp.json()
    assert snap["map"]["width"] == 32 and snap["map"]["height"] == 24
    assert {a["name"] for a in snap["actors"]} == {"민아", "준호", "서연", "도윤", "하늘"}
    assert len(snap["objects"]) >= 40 and "mina" in snap["relationships"]
    assert all("missing" not in o["data"] for o in snap["objects"])
    agent = client.get("/api/agent/mina").json()
    assert agent["name"] == "민아" and set(agent["needs"]) == {"hunger", "stamina", "mood"}
    assert set(agent["relationships"]["junho"]) >= {"familiarity", "affection", "trust",
                                                    "respect"}
    assert client.get("/api/agent/nobody").status_code == 404


def test_ws_snapshot_then_deltas_on_step(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot" and first["clock"]["paused"]
        ws.send_text(json.dumps({"cmd": "step", "ticks": 120}))  # 05:00 -> 07:00
        msgs = collect(ws, lambda m: m["type"] == "batch" and m["tick"] >= first["clock"]["tick"]
                       + 120)
        acks = [m for m in msgs if m["type"] == "ack"]
        assert acks and acks[0]["ok"]
        types = {d["type"] for d in deltas(msgs)}
        assert {"clock", "agent_state", "path_move"} <= types
        agent_state = next(d for d in deltas(msgs) if d["type"] == "agent_state")
        assert {"actor_id", "x", "y", "facing", "activity", "hunger"} <= set(agent_state)
    assert client.get("/api/health").json()["minute"] >= 300 + 120


def test_pause_resume_speed_and_god_commands(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        def cmd(payload):
            ws.send_text(json.dumps(payload))
            msgs = collect(ws, lambda m: m["type"] == "ack")
            return msgs[-1], msgs

        ack, _ = cmd({"cmd": "set_speed", "speed": 4})
        assert ack["ok"] and client.get("/api/health").json()["speed"] == 4.0
        bad, _ = cmd({"cmd": "set_speed", "speed": 3})
        assert not bad["ok"]
        ack, msgs = cmd({"cmd": "god_weather", "weather": "Rain"})
        assert ack["ok"]
        assert any(d["type"] == "god_event" and d["kind"] == "set_weather" for d in deltas(msgs))
        assert client.get("/api/snapshot").json()["clock"]["weather"] == "rain"
        ack, msgs = cmd({"cmd": "god_spawn", "kind": "berry_bush", "x": 12, "y": 16})
        assert ack["ok"]
        assert any(d["type"] == "object_change" and d["object_id"].startswith("berry_bush_spawn")
                   for d in deltas(msgs))
        assert not cmd({"cmd": "god_spawn", "kind": "berry_bush", "x": 22, "y": 13})[0]["ok"]
        ack, msgs = cmd({"cmd": "god_event", "text": "광장에 축제가 열렸다!"})
        assert ack["ok"]
        assert any(d["type"] == "event" and d["kind"] == "god_inject_event"
                   for d in deltas(msgs))
        assert cmd({"cmd": "god_rumor", "target": "junho", "about": "doyun",
                    "text": "도윤이 창고를 뒤졌대"})[0]["ok"]
        assert client.get("/api/agent/junho").json()["relationships"]["doyun"]["trust"] < 0
        assert cmd({"cmd": "checkpoint"})[0]["ok"]
        assert not cmd({"cmd": "warp"})[0]["ok"]
        ws.send_text("not json")
        assert not collect(ws, lambda m: m["type"] == "ack")[-1]["ok"]
        ack, msgs = cmd({"cmd": "resume"})
        assert ack["ok"]
        collect(ws, lambda m: m["type"] == "batch" and any(
            d["type"] == "clock" and not d["paused"] for d in m["deltas"]))
        assert cmd({"cmd": "pause"})[0]["ok"]
        assert client.get("/api/health").json()["paused"]


# ---------------------------------------------------------------- review regressions
def test_paused_manual_step_is_not_throttled_by_speed(config, tmp_path):
    import time

    app = create_app(config, speed=0.5, paused=True, t_max_seconds=0.0)
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        t0 = time.perf_counter()
        ws.send_text(json.dumps({"cmd": "step", "ticks": 120}))
        msgs = collect(ws, lambda m: m["type"] == "batch"
                       and m["tick"] >= first["clock"]["tick"] + 120, limit=2000)
        # at 0.5x pacing 120 ticks would take 120 s; manual steps bypass pacing
        assert time.perf_counter() - t0 < 10
        moves = [d for d in deltas(msgs) if d["type"] == "path_move"]
        assert moves and all({"x", "y"} == set(d["start_pos"]) for d in moves)


def test_ws_rejects_foreign_origin(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5174"}) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_agent_detail_has_incoming_edges(client):
    d = client.get("/api/agent/mina").json()
    assert set(d["incoming"]) == {"junho", "seoyeon", "doyun", "haneul"}
    assert "affection" in d["incoming"]["junho"]


async def test_slow_client_is_closed_not_silently_dropped(config):
    from llm_vil.engine import SimEngine
    from llm_vil.llm import DemoBackend
    from llm_vil.server.runtime import MAX_CLIENT_BACKLOG, SimRuntime

    rt = SimRuntime(SimEngine(config, DemoBackend()), paused=True)
    q = rt.subscribe()
    for _ in range(MAX_CLIENT_BACKLOG + 1):
        q.put_nowait({"type": "batch"})
    rt._emit({"type": "clock"})
    rt._flush(force=True)
    assert q not in rt.clients and rt.dropped_clients == 1
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert items[-1] is None  # close sentinel for the socket pump


async def test_commands_serialize_with_steps(config):
    import asyncio

    from llm_vil.engine import SimEngine
    from llm_vil.llm import DemoBackend
    from llm_vil.server.runtime import SimRuntime

    e = SimEngine(config, DemoBackend())
    rt = SimRuntime(e, paused=True)
    jobs = []
    for i in range(30):
        jobs.append(rt.tick_once())
        jobs.append(rt.command({"cmd": "god_event", "text": f"사건 {i}"} if i % 3 == 0
                               else {"cmd": "god_weather", "weather": "rain"}))
    results = await asyncio.gather(*jobs)
    assert all(r is None or r["ok"] for r in results)
    assert e.metrics["errors"] == 0 and e.metrics["god_events"] == 30


def test_llm_stats_endpoint_has_no_secrets(config, monkeypatch):
    secret = "sk-live-NEVER-SHOW-ME-42"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    app = create_app(config, speed=8.0, paused=True, t_max_seconds=0.0)
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        ws.send_text(json.dumps({"cmd": "step", "ticks": 90}))
        collect(ws, lambda m: m["type"] == "batch" and m["tick"] >= first["clock"]["tick"] + 90)
        resp = c.get("/api/llm/stats")
        s = resp.json()
        assert s["profile"] == "demo" and s["provider"] == "demo"
        assert s["embed_provider"] == "demo" and s["embed_cache_misses"] > 0
        assert s["total_calls"] > 0 and s["validity_rate"] == 1.0
        assert set(s["latency_ms"]) == {"p50", "p95", "max"}
        assert s["recent_calls"] and {"tick", "agent_id", "purpose", "tier", "latency_ms",
                                      "status"} <= set(s["recent_calls"][0])
        assert set(s["per_agent"]) <= {"mina", "junho", "seoyeon", "doyun", "haneul"}
        for body in (resp.text, json.dumps(first, ensure_ascii=False)):
            assert secret not in body and "api_key\"" not in body


def test_openai_profile_stats_hide_key(config, monkeypatch):
    secret = "sk-live-NEVER-SHOW-ME-43"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    app = create_app(config, speed=8.0, paused=True, llm_profile="openai_compat",
                     embed_profile="demo")
    with TestClient(app) as c:
        s = c.get("/api/llm/stats")
        assert s.json()["provider"] == "openai_compat" and s.json()["fast_model"]
        assert secret not in s.text

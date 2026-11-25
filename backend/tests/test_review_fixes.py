"""Regression tests for the M1 review findings (7 Major + key Minor)."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from llm_vil.agent import Agent
from llm_vil.agent.memory import MemoryStream
from llm_vil.agent.replies import PlanBlockReply, ReflectReply
from llm_vil.engine import CognitionClock, SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import (
    HashingEmbedder,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    dumps,
    strip_think,
)
from llm_vil.schemas.agent import MemoryRecord
from llm_vil.schemas.world import AtomicAction
from llm_vil.world.world import World

DAY = 1440


class GatedLLM:
    model = "gated"

    def __init__(self, reply: dict):
        self.reply = reply
        self.release = asyncio.Event()

    async def complete_json(self, req):
        await self.release.wait()
        return LLMResponse(text=dumps(self.reply), model=self.model)

    async def embed(self, texts):
        return [[0.0] for _ in texts]


class ScriptedLLM:
    model = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def complete_json(self, req):
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=self.model)

    async def embed(self, texts):
        return [[0.0] for _ in texts]


def make_agent(config, aid="mina", llm=None) -> Agent:
    spec = next(v for v in config.villagers if v.id == aid)
    return Agent(spec, config, LLMRouter(llm or DemoBackend(), retry_base_delay_s=0),
                 HashingEmbedder())


# ---------------------------------------------------------------- 1. dual-trigger reflection
async def test_importance_and_end_of_day_reflections_both_fire(config):
    cfg = config.model_copy(update={"memory": config.memory.model_copy(
        update={"reflection_threshold": 40})})
    e = SimEngine(cfg, DemoBackend())
    await e.run_days(1)
    assert e.metrics["threshold_reflections"] > 0
    both = [a for a in e.agents.values()
            if a.last_threshold_reflection_tick // DAY == a.last_reflection_day >= 0]
    assert both, "an agent should reflect on importance AND at end of day on the same day"
    assert e.metrics["reflections"] > e.metrics["threshold_reflections"]


def test_reflection_bookkeeping(config):
    agent = make_agent(config)
    w = World(config)
    view = w.perceive("mina")
    assert agent.adopt_reflection(None, view, mode="importance") is False  # None never counts
    rec = agent.memory.add(view.tick, "observation", "큰 일", 8)
    reply = ReflectReply.model_validate({
        "reflections": [{"text": "깨달음", "importance": 7, "evidence_ids": [rec.id]}],
        "facts": ["사실 하나", "사실 둘"],
    })
    assert agent.adopt_reflection(reply, view, mode="importance") is True
    assert agent.memory.importance_accum == 0  # facts do not re-trigger reflection
    assert agent.last_reflection_day == -1  # importance mode leaves end-of-day slot free
    assert agent.wants_weekly(7) and not agent.adopt_weekly(None, view)
    agent.last_weekly_day = 6
    assert not agent.wants_weekly(12) and agent.wants_weekly(13)  # robust past midnight


# ---------------------------------------------------------------- 2. automatic checkpoints
async def test_automatic_checkpoint_with_awake_villagers_is_bit_identical(config, tmp_path):
    night_owls = [v.model_copy(update={"sleep_minute": 1439}) for v in config.villagers]
    cfg = config.model_copy(update={"villagers": night_owls})
    path = tmp_path / "auto.sqlite"
    end = 2 * DAY + 300
    store = SaveStore(path)
    a = SimEngine(cfg, DemoBackend(), store=store, checkpoint_every_days=1)
    await a.run_until(end)
    a.flush()
    snap_a = dumps(a.snapshot())
    tick, data = store.load_checkpoint(2 * DAY + 60)
    assert tick >= 2 * DAY
    assert any(act["activity"] != "sleeping" for act in data["world"]["actors"])
    store.close()
    store_b = SaveStore(path)
    b = SimEngine.from_checkpoint(cfg, DemoBackend(), store_b, at_or_before=tick,
                                  checkpoint_every_days=1)
    await b.run_until(end)
    b.flush()
    assert dumps(b.snapshot()) == snap_a
    store_b.close()


async def test_pending_checkpoint_waits_for_quiescence(config, tmp_path):
    store = SaveStore(tmp_path / "p.sqlite")
    llm = GatedLLM({"routine": "rest", "target": "home_mina"})
    e = SimEngine(config, llm, barrier_mode=False, store=store,
                  clock=CognitionClock(t_max_seconds=60.0), checkpoint_every_days=1)
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "test")
    e.world.tick = DAY - 1
    await e.step()  # midnight with cognition in flight
    assert e.checkpoint_pending and store.count("checkpoints") == 0
    llm.release.set()
    await asyncio.sleep(0.01)
    await e.step()
    assert not e.checkpoint_pending and store.count("checkpoints") == 1


# ---------------------------------------------------------------- 3. stale view & merchant
async def test_stale_validation_covers_known_but_unseen_targets(config):
    llm = GatedLLM({"routine": "tend_farm", "target": "my_plots"})
    e = SimEngine(config, llm, barrier_mode=False, clock=CognitionClock(t_max_seconds=60.0))
    e.world.apply("mina", AtomicAction(kind="wake"))
    view = e.world.perceive("mina")
    assert "plot_09" not in {o.id for o in view.objects}  # far outside perception radius
    e._spawn("mina", "choose_routine", "test")
    await asyncio.sleep(0)
    e.world._set_state(e.world.objects["plot_09"], "tilled")
    llm.release.set()
    await e.jobs["mina"].task
    e._harvest()
    assert e.metrics["stale_dropped"] == 1
    assert e.triggers["mina"]["choose_routine"] == "stale_view:plot_09"


async def test_merchant_toggle_bumps_stall(config):
    e = SimEngine(config, DemoBackend())
    stall = e.world.objects["stall_main"]
    v0, wv0 = stall.version, e.world.world_version
    await e.run_until(config.economy.merchant_open_minute)
    assert stall.state == "merchant_open" and stall.version > v0
    assert e.world.world_version > wv0
    assert any(d.type == "object_change" and d.object_id == "stall_main"
               and d.state == "merchant_open" for d in e.drain_deltas())
    await e.run_until(config.economy.merchant_close_minute)
    assert stall.state == "idle"


# ---------------------------------------------------------------- 4. cancellation & God events
async def test_inject_event_cancels_cleanly_and_is_logged(config, tmp_path):
    store = SaveStore(tmp_path / "g.sqlite")
    llm = GatedLLM({"routine": "rest", "target": "home_mina"})
    fake_now = [100.0]
    clock = CognitionClock(t_max_seconds=60.0, time_fn=lambda: fake_now[0])
    e = SimEngine(config, llm, barrier_mode=False, store=store, clock=clock)
    for aid in ("mina", "junho"):
        e.world.apply(aid, AtomicAction(kind="wake"))
    e.world.apply("junho", AtomicAction(kind="move_to", target_id="tree_05"))
    e._spawn("mina", "choose_routine", "test")
    assert e.jobs["mina"].started == 100.0  # injected clock, not time.monotonic()
    await asyncio.sleep(0)  # mina's task now holds a SharedRateGate slot
    assert e.router.gate.in_flight == 1
    e.inject_event("산사태가 났다!")
    assert "mina" not in e.jobs and e.metrics["cancelled_jobs"] == 1
    assert e.triggers["mina"] == {"choose_routine": "external_event"}
    assert e.world.actors["junho"].action is None  # durative action interrupted
    assert all("turn" not in t for t in e.triggers.values())
    await e.quiesce()
    assert e._cancelled == [] and e.router.gate.in_flight == 0
    e.set_weather("rain")
    assert any("weather_changed" in r.text for r in e.agents["haneul"].memory.records)
    e.inject_rumor("junho", "doyun", "도윤이 창고를 뒤졌대")
    e.flush()
    kinds = {r[0] for r in store.conn.execute("SELECT kind FROM events")}
    assert {"god_inject_event", "god_set_weather", "god_inject_rumor", "weather_changed"} <= kinds
    store.close()


# ---------------------------------------------------------------- 5. archive embeddings & store
def test_archived_embeddings_give_real_relevance(tmp_path):
    emb = HashingEmbedder()
    store = SaveStore(tmp_path / "e.sqlite")
    from llm_vil.schemas.config import MemoryConfig

    m = MemoryStream("mina", emb, MemoryConfig())
    hit = m.add(0, "observation", "준호와 장작 이야기를 했다", 3, subjects=["junho"])
    miss = m.add(0, "observation", "산딸기가 달았다", 3, subjects=["junho"])
    store.write_batch(archive=[(hit, 10), (miss, 10)])
    m.records = []
    archived = store.search_archive("mina", subject="junho", limit=2)
    assert all(r.embedding and r.embed_model == emb.model for r in archived)
    top = m.retrieve(20, "장작 이야기", extra=archived, k=1)
    assert top[0].id == hit.id
    assert store.count("archive_subjects") == 2
    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT memory_id FROM archive_subjects WHERE agent_id=? AND subject=?",
        ("mina", "junho")).fetchall()
    assert "subject=?" in str(plan) and "SCAN" not in str(plan)  # indexed, no LIKE scan


def test_migration_v3_backfills_subject_index(tmp_path):
    path = tmp_path / "old.sqlite"
    s = SaveStore(path)
    rec = MemoryRecord(id="mina-m00001", owner_id="mina", tick=0, kind="observation",
                       text="옛 기억", importance=3, subjects=["junho", "seoyeon"])
    s.write_batch(archive=[(rec, 5)])
    s.conn.execute("DROP TABLE archive_subjects")
    s.conn.execute("ALTER TABLE tape DROP COLUMN error")
    s.conn.execute("DROP TABLE embed_cache")
    s.conn.execute("DROP INDEX idx_archive_embed")
    s.conn.execute("ALTER TABLE memory_archive DROP COLUMN embed_model")
    s.conn.execute("ALTER TABLE memory_archive DROP COLUMN embed_dim")
    s.conn.execute("ALTER TABLE checkpoints DROP COLUMN label")
    s.conn.execute("DROP TABLE replay_frames")
    s.conn.execute("DELETE FROM schema_version WHERE version >= 3")
    s.close()
    s2 = SaveStore(path)
    from llm_vil.engine.save_store import SCHEMA_VERSION

    assert s2.schema_version() == SCHEMA_VERSION
    assert [r.id for r in s2.search_archive("mina", subject="seoyeon")] == ["mina-m00001"]
    s2.close()


def test_truncate_after_rolls_back_on_failure(tmp_path):
    s = SaveStore(tmp_path / "t.sqlite")
    s.save_checkpoint(10, "{}")
    s.save_checkpoint(20, "{}")
    real = s.conn

    class Failing:
        def execute(self, sql, *args):
            if sql.startswith("DELETE FROM tape"):
                raise RuntimeError("disk full")
            return real.execute(sql, *args)

    s.conn = Failing()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        s.truncate_after(10)
    s.conn = real
    assert not real.in_transaction
    assert s.count("checkpoints") == 2  # nothing partially deleted


# ---------------------------------------------------------------- 6. information boundary
def test_world_view_never_leaks_thief_identity(config):
    w = World(config)
    for aid in w.actors:
        w.apply(aid, AtomicAction(kind="wake"))
    chest = w.objects["chest_haneul"]
    w.actors["doyun"].pos = (chest.pos[0], chest.pos[1] + 1)
    for _ in range(chest.data["items"]["berry"]):  # empty the chest completely
        assert w.apply("doyun", AtomicAction(kind="take", target_id="chest_haneul",
                                             item="berry")).ok
        while w.actors["doyun"].action is not None:
            w.tick_once()
    assert chest.data["missing"]
    view = w.perceive("doyun")
    seen = next(o for o in view.objects if o.id == "chest_haneul")
    assert "missing" not in seen.data and "taker" not in dumps(seen.data)
    w.actors["haneul"].pos = (chest.pos[0] + 1, chest.pos[1] + 1)
    res = w.apply("haneul", AtomicAction(kind="take", target_id="chest_haneul", item="berry"))
    assert res.reason == "chest_empty"
    assert [e.kind for e in res.events] == ["chest_missing_discovered"]
    assert chest.data["missing"] == []


def test_initial_known_world_hides_other_personal_chests(config):
    agent = make_agent(config, "mina")
    assert agent.known["chest_mina"].items == {"berry": 2}
    assert agent.known["chest_shared"].items
    assert agent.known["chest_haneul"].items == {}
    assert agent.known["chest_doyun"].items == {}


async def test_owner_discovery_on_empty_chest_reaches_memory(config):
    e = SimEngine(config, DemoBackend())
    w = e.world
    for aid in w.actors:
        w.apply(aid, AtomicAction(kind="wake"))
    chest = w.objects["chest_haneul"]
    chest.data["items"] = {}
    chest.data["missing"] = [{"taker": "doyun", "item": "berry", "qty": 3, "tick": 0}]
    w.actors["haneul"].pos = (chest.pos[0] + 1, chest.pos[1] + 1)
    from llm_vil.schemas.agent import RoutineChoice

    agent = e.agents["haneul"]
    agent.routine = RoutineChoice(routine="store_or_take", target="chest_haneul", item="berry",
                                  op="take")
    e._act("haneul")
    assert any("사라졌다" in r.text for r in agent.memory.records)


# ---------------------------------------------------------------- 7. validation & backoff
async def test_invalid_chest_op_and_qty_are_rejected(config):
    bad = '{"routine": "store_or_take", "target": "chest_mina", "item": "berry", "op": "none"}'
    greedy = ('{"routine": "store_or_take", "target": "chest_mina", "item": "berry", '
              '"op": "take", "qty": 9}')
    good = ('{"routine": "store_or_take", "target": "chest_mina", "item": "berry", '
            '"op": "take", "qty": 2}')
    llm = ScriptedLLM([bad, greedy, good])
    agent = make_agent(config, "mina", llm)
    w = World(config)
    w.apply("mina", AtomicAction(kind="wake"))
    dec = await agent.decide_routine(w.perceive("mina"), "test")
    assert not dec.fallback and dec.choice.op == "take" and dec.choice.qty == 2
    assert agent.router.stats.calls == 3
    fb = make_agent(config, "mina", ScriptedLLM([bad] * 3))
    dec2 = await fb.decide_routine(w.perceive("mina"), "test")
    assert dec2.fallback and dec2.choice.routine != "store_or_take"


async def test_trade_validation_checks_coins(config):
    agent = make_agent(config, "mina")
    from llm_vil.agent.replies import RoutineReply

    r = RoutineReply(routine="trade", target="stall_main", item="bread", op="buy", qty=5)
    assert "coins" in agent._validate_trade(r, {"coins": 10})
    r_ok = RoutineReply(routine="trade", target="stall_main", item="bread", op="buy", qty=1)
    assert agent._validate_trade(r_ok, {"coins": 10}) is None
    assert "op must be" in agent._validate_trade(
        RoutineReply(routine="trade", target="stall_main", item="bread"), {"coins": 10})


async def test_provider_errors_back_off_and_are_taped():
    llm = ScriptedLLM([RuntimeError("503"), RuntimeError("503"), '{"routine": "rest"}'])
    router = LLMRouter(llm, retries=2, retry_base_delay_s=0.01)
    from llm_vil.agent.replies import RoutineReply

    t0 = time.perf_counter()
    out = await router.call(LLMRequest(purpose="choose_routine", agent_id="mina", tick=0),
                            RoutineReply)
    assert out is not None and out.routine == "rest"
    assert time.perf_counter() - t0 >= 0.03  # 0.01 + 0.02 exponential backoff
    tape = router.drain_tape()
    assert [t.ok for t in tape] == [False, False, True]
    assert tape[0].error.startswith("RuntimeError") and router.stats.errors == 2


async def test_timeouts_are_taped():
    class Slow(ScriptedLLM):
        async def complete_json(self, req):
            await asyncio.sleep(1)

    router = LLMRouter(Slow([]), timeout=0.01, retry_base_delay_s=0)
    from llm_vil.agent.replies import RoutineReply

    assert await router.call(LLMRequest(purpose="x", agent_id="a", tick=0), RoutineReply) is None
    assert router.drain_tape()[0].error.startswith("timeout")


# ---------------------------------------------------------------- minors
def test_strip_think_handles_unclosed_and_orphan_tags():
    assert strip_think('{"a": 1}<think>truncated reasoning') == '{"a": 1}'
    assert strip_think('reasoning only</think>{"a": 1}') == '{"a": 1}'
    assert strip_think('<think>x</think>{"a": 1}') == '{"a": 1}'


def test_plan_block_requires_positive_span():
    with pytest.raises(ValidationError):
        PlanBlockReply(start_minute=600, end_minute=600, routine="rest")
    assert PlanBlockReply(start_minute=600, end_minute=660, routine="rest")


async def test_load_snapshot_refuses_inflight_and_resets_buffers(config):
    llm = GatedLLM({"routine": "rest", "target": "home_mina"})
    e = SimEngine(config, llm, barrier_mode=False, clock=CognitionClock(t_max_seconds=60.0))
    snap = SimEngine(config, DemoBackend()).snapshot()
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "x")
    with pytest.raises(RuntimeError):
        e.load_snapshot(snap)
    llm.release.set()
    await e.quiesce()
    e.errors.append("old")
    e.pending_events.append(e.world._event("x"))
    e.load_snapshot(snap)
    assert e.errors == [] and e.pending_events == [] and not e.deltas

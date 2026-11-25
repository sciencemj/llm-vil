"""Async latency race tests with injected DemoBackend latency."""

import asyncio

from llm_vil.engine import CognitionClock, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import LLMResponse, dumps
from llm_vil.schemas.world import AtomicAction


async def test_latency_race_dilates_clock_without_errors(config):
    llm = DemoBackend(simulated_latency_ms=40)
    clock = CognitionClock(speed=250.0, sim_minutes_per_real_second=2.0, t_max_seconds=0.02)
    e = SimEngine(config, llm, barrier_mode=False, clock=clock, max_concurrency=3)
    await e.run_until(config.start_minute + 8 * 60)  # 05:00 -> 13:00 with ~40ms cognition
    # interrupt in-flight intents with a God event while cognition is pending
    for _ in range(500):
        if e.jobs:
            break
        await e.step()
    assert e.jobs, "expected in-flight cognition under latency"
    e.inject_event("마을 광장에 커다란 무지개가 떴다!")
    assert e.metrics["cancelled_jobs"] >= 1
    await e.run_until(e.tick + 120)
    await e.quiesce()

    s = e.summary()
    assert s["errors"] == [] and e.metrics["errors"] == 0
    assert clock.dilated_ticks > 0 and e.metrics["dilated_ticks"] > 0
    assert clock.max_cognition_wait >= 0.02
    assert e.router.stats.fallbacks == 0 and e.router.stats.timeouts == 0
    assert e.router.gate.peak <= 3
    assert e.metrics["cognition_jobs"] > 20
    mt = s["metrics"]
    assert mt["wood_chopped"] + mt["berries_foraged"] + mt["orders_fulfilled"] > 0
    assert all("마을 광장에 커다란 무지개" in r.text
               for a in e.agents.values() for r in a.memory.of_kind("external_event")[-1:])
    dumps(e.snapshot())  # quiesced engine is snapshot-able


class GatedLLM:
    """Answers a fixed routine only after ``release`` is set."""

    model = "gated"

    def __init__(self, reply: dict):
        self.reply = reply
        self.release = asyncio.Event()

    async def complete_json(self, req):
        await self.release.wait()
        return LLMResponse(text=dumps(self.reply), model=self.model)

    async def embed(self, texts):
        return [[0.0] for _ in texts]


async def test_stale_intent_is_dropped_and_requeued(config):
    llm = GatedLLM({"routine": "forage", "target": "bush_nw"})
    e = SimEngine(config, llm, barrier_mode=False, clock=CognitionClock(t_max_seconds=5.0))
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "test")
    await asyncio.sleep(0)  # cognition is now in flight with a view of ripe bush_nw
    bush = e.world.objects["bush_nw"]
    e.world._set_state(bush, "empty")  # salient change invalidates the view
    llm.release.set()
    await e.jobs["mina"].task
    e._harvest()
    assert e.metrics["stale_dropped"] == 1
    assert e.agents["mina"].routine is None
    assert e.triggers["mina"]["choose_routine"] == "stale_view:bush_nw"


async def test_fresh_intent_is_adopted(config):
    llm = GatedLLM({"routine": "forage", "target": "bush_nw"})
    e = SimEngine(config, llm, barrier_mode=False, clock=CognitionClock(t_max_seconds=5.0))
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "test")
    llm.release.set()
    await e.jobs["mina"].task
    e._harvest()
    assert e.metrics["stale_dropped"] == 0
    assert e.agents["mina"].routine is not None
    assert e.world.actors["mina"].activity == "walking"  # executor started immediately


async def test_one_inflight_cognition_per_agent(config):
    llm = GatedLLM({"routine": "rest", "target": "home_mina"})
    e = SimEngine(config, llm, barrier_mode=False, clock=CognitionClock(t_max_seconds=5.0))
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "first")
    first = e.jobs["mina"].task
    e._trigger("mina", "choose_routine", "second")
    e._drive_agents()  # must not spawn a second task while one is in flight
    assert e.jobs["mina"].task is first
    assert e.triggers["mina"] == {"choose_routine": "second"}
    llm.release.set()
    await e.quiesce()

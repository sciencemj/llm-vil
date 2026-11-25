"""Decision barrier: under real (staggered) LLM latency every villager's decision lands on
the same sim tick, so the live view shows them start moving together."""

from __future__ import annotations

import asyncio

from llm_vil.engine import CognitionClock, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.schemas.protocol import PathMoveDelta
from llm_vil.schemas.world import AtomicAction


class StaggeredDemo(DemoBackend):
    """DemoBackend whose answer time depends on the villager (5 ms ... 50 ms)."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.latency = {aid: 0.005 * (i + 1) for i, aid in enumerate(order)}

    async def complete_json(self, req):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.latency.get(req.agent_id, 0.005))
        return await super().complete_json(req)


def live_engine(config, *, barrier: bool) -> SimEngine:
    # live (non-deterministic) clock: t_max > 0 lets the sim run ahead of slow cognition
    llm = StaggeredDemo([v.id for v in config.villagers])
    e = SimEngine(config, llm, clock=CognitionClock(t_max_seconds=5.0), barrier_mode=barrier)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    return e


def record_adoptions(e: SimEngine) -> dict[str, list[int]]:
    """Tick at which each villager adopted each routine decision."""
    seen: dict[str, list[int]] = {aid: [] for aid in e.order}
    for aid, agent in e.agents.items():
        orig = agent.adopt_routine

        def wrapped(decision, _aid=aid, _orig=orig):  # type: ignore[no-untyped-def]
            seen[_aid].append(e.world.tick)
            return _orig(decision)

        agent.adopt_routine = wrapped  # type: ignore[method-assign]
    return seen


async def run_until_all_decided(e: SimEngine, seen: dict[str, list[int]],
                                max_steps: int = 2000) -> None:
    for _ in range(max_steps):
        if all(seen.values()):
            return
        await e.step()
    raise AssertionError(f"not everyone decided: {seen}")


async def test_barrier_applies_all_decisions_on_the_same_tick(config):
    e = live_engine(config, barrier=True)
    assert e.barrier_active
    seen = record_adoptions(e)
    start = e.world.tick
    await run_until_all_decided(e, seen)
    first = {aid: ticks[0] for aid, ticks in seen.items()}
    assert len(set(first.values())) == 1, first  # every first decision on one tick
    assert e.world.tick - start <= 3  # the tick was held, not run ahead of cognition
    assert e.metrics["barrier_waits"] >= 1 and e.metrics["barrier_timeouts"] == 0
    assert e.metrics["barrier_max_batch"] == len(e.order)
    # the resulting walks leave in one synchronized batch: same start tick for everyone
    moves = [d for d in e.drain_deltas() if isinstance(d, PathMoveDelta)]
    starts = {}
    for d in moves:
        starts.setdefault(d.actor_id, d.start_tick)
    assert starts and len(set(starts.values())) == 1, starts
    assert e.metrics["errors"] == 0


async def test_without_barrier_decisions_trickle_in_over_several_ticks(config):
    e = live_engine(config, barrier=False)
    assert not e.barrier_active
    seen = record_adoptions(e)
    await run_until_all_decided(e, seen)
    first = {aid: ticks[0] for aid, ticks in seen.items()}
    assert len(set(first.values())) > 1, first  # the old, one-by-one behaviour
    assert e.metrics["barrier_waits"] == 0


async def test_barrier_holds_tick_and_reports_thinking_actors(config):
    e = live_engine(config, barrier=True)
    e.barrier_slice_s = 0.001  # one short slice per step: observe the held state
    tick0 = e.world.tick
    held = False
    for _ in range(50):
        advanced = await e.step()
        if not advanced:
            held = True
            assert e.world.tick == tick0 or e.thinking_actors  # frozen while thinking
            assert e.thinking_actors == [a for a in e.order if a in e.jobs]
            assert len(e.thinking_actors) >= 1
            break
        tick0 = e.world.tick
    assert held


async def test_barrier_times_out_instead_of_deadlocking(config):
    class Stuck(DemoBackend):
        async def complete_json(self, req):  # type: ignore[no-untyped-def]
            if req.agent_id == "mina":
                await asyncio.Event().wait()  # never answers
            return await super().complete_json(req)

    e = SimEngine(config, Stuck(), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True, barrier_timeout_s=0.05, llm_timeout=30.0)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    tick0 = e.world.tick
    for _ in range(400):
        await e.step()
        if e.world.tick >= tick0 + 5:
            break
    assert e.world.tick >= tick0 + 5  # the world moved on without mina's answer
    assert e.metrics["barrier_timeouts"] >= 1
    assert e.jobs["mina"].timed_out and "mina" not in e.thinking_actors
    for job in list(e.jobs.values()):
        job.task.cancel()
    await asyncio.gather(*(j.task for j in e.jobs.values()), return_exceptions=True)


async def test_runtime_streams_thinking_status_and_releases_in_one_batch(config):
    from llm_vil.server.runtime import SimRuntime

    e = live_engine(config, barrier=True)
    rt = SimRuntime(e, paused=True)
    q: asyncio.Queue = asyncio.Queue()
    rt.clients.add(q)
    thinking_seen = False
    for _ in range(400):
        await rt.tick_once()
        if rt.health()["thinking_count"] > 0:
            thinking_seen = True
            assert rt.health()["barrier_mode"] is True
        if thinking_seen and not e.jobs:
            break
    rt._flush(force=True)
    batches = []
    while not q.empty():
        m = q.get_nowait()
        if m and m.get("type") == "batch":
            batches.append(m)
    clocks = [d for b in batches for d in b["deltas"] if d["type"] == "clock"]
    assert thinking_seen and any(c.get("thinking_count", 0) > 0 for c in clocks)
    assert all("thinking_actors" in c for c in clocks)
    # every first walk of the release sits in a single batch
    first_batch = {}
    for i, b in enumerate(batches):
        for d in b["deltas"]:
            if d["type"] == "path_move":
                first_batch.setdefault(d["actor_id"], i)
    assert first_batch and len(set(first_batch.values())) == 1, first_batch


class SlowDemo(DemoBackend):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def complete_json(self, req):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        return await super().complete_json(req)


async def step_until_held(e: SimEngine, limit: int = 400) -> None:
    for _ in range(limit):
        if not await e.step():
            return
    raise AssertionError("barrier never held a tick")


async def test_hung_job_blocks_only_once_then_ticks_flow(config):
    class Stuck(DemoBackend):
        async def complete_json(self, req):  # type: ignore[no-untyped-def]
            if req.agent_id == "mina":
                await asyncio.Event().wait()
            return await super().complete_json(req)

    e = SimEngine(config, Stuck(), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True, barrier_timeout_s=0.05)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    e.barrier_slice_s = 0.01
    await step_until_held(e)
    while e.metrics["barrier_timeouts"] == 0:
        await e.step()
    assert e.jobs["mina"].timed_out
    # the hung call never re-blocks: the next 60 ticks advance without another timeout
    tick0 = e.world.tick
    for _ in range(400):
        await e.step()
        if e.world.tick >= tick0 + 60:
            break
    assert e.world.tick >= tick0 + 60
    assert e.metrics["barrier_timeouts"] == 1
    assert "mina" in e.jobs and "mina" not in e.thinking_actors
    e.jobs["mina"].task.cancel()
    await asyncio.gather(e.jobs["mina"].task, return_exceptions=True)


async def test_thinking_actors_count_down_while_held(config):
    e = live_engine(config, barrier=True)
    e.barrier_slice_s = 0.002
    await step_until_held(e)
    counts = [len(e.thinking_actors)]
    while not await e.step():
        counts.append(len(e.thinking_actors))
        assert all(not e.jobs[a].task.done() for a in e.thinking_actors)
    assert counts[0] >= 3 and counts == sorted(counts, reverse=True), counts
    assert len(set(counts)) >= 3  # 10 -> 9 -> ... as answers arrive, not a frozen number


async def test_cancelling_every_job_clears_the_barrier_clock(config):
    e = SimEngine(config, SlowDemo(5.0), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    e.barrier_slice_s = 0.001
    await step_until_held(e)
    assert e._barrier_since is not None
    for aid in list(e.jobs):
        e._cancel_job(aid, ("choose_routine", "plan_day", "reflect", "reflect_threshold",
                            "weekly", "turn"))
    assert e._barrier_since is None and e.thinking_actors == []
    await e.quiesce()
    assert e._barrier_since is None


async def test_switch_llm_mid_barrier_resets_hold_and_recomputes_timeout(config):
    from llm_vil.server.runtime import SimRuntime

    e = SimEngine(config, SlowDemo(5.0), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True, llm_timeout=30.0, json_repair_retries=2)
    assert e.barrier_timeout_s == 30.0 * 4
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    rt = SimRuntime(e, paused=True)
    e.barrier_slice_s = 0.01
    while await rt.tick_once():
        pass
    assert rt._held and e._barrier_since is not None
    await asyncio.sleep(0.6)  # a stale wait start would eat the next wave's whole budget
    await rt.switch_llm(SlowDemo(0.15), e.embedder, timeout=0.4, retries=0)
    assert e.barrier_timeout_s == 0.4 * 2  # recomputed from the new profile (was 120 s)
    assert not rt._held and e._barrier_since is None
    # the next wave (0.15 s answers) completes inside the fresh 0.8 s budget
    for _ in range(400):
        await rt.tick_once()
        if e.metrics["barrier_batches"] >= 1 and not e.jobs:
            break
    assert e.metrics["barrier_timeouts"] == 0 and e.metrics["errors"] == 0


async def test_god_event_mid_barrier_keeps_deadline_and_streams_clock(config):
    from llm_vil.server.runtime import SimRuntime

    e = SimEngine(config, SlowDemo(0.2), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True, barrier_timeout_s=2.0)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    rt = SimRuntime(e, paused=True)
    q: asyncio.Queue = asyncio.Queue()
    rt.clients.add(q)
    e.barrier_slice_s = 0.01
    while await rt.tick_once():
        pass
    ack = await rt.command({"cmd": "god_event", "text": "광장에 축제가 열렸다!"})
    assert ack["ok"]
    msgs = []
    while not q.empty():
        m = q.get_nowait()
        if m and m.get("type") == "batch":
            msgs.extend(m["deltas"])
    clocks = [d for d in msgs if d["type"] == "clock"]
    assert clocks and "thinking_count" in clocks[-1]
    for _ in range(600):  # every wave after the event still settles inside its budget
        await rt.tick_once()
        if e.metrics["barrier_batches"] >= 3:
            break
    assert e.metrics["barrier_batches"] >= 3 and e.metrics["barrier_timeouts"] == 0


async def test_snapshot_clock_reports_held_state(config):
    from llm_vil.server.runtime import SimRuntime

    e = SimEngine(config, SlowDemo(5.0), clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    rt = SimRuntime(e, paused=True)
    e.barrier_slice_s = 0.001
    while await rt.tick_once():
        pass
    clock = rt.snapshot()["clock"]
    assert clock["thinking_count"] == len(e.thinking_actors) > 0
    assert clock["thinking_actors"] == e.thinking_actors
    for job in list(e.jobs.values()):
        job.task.cancel()
    await asyncio.gather(*(j.task for j in e.jobs.values()), return_exceptions=True)


async def test_save_checkpoint_mid_barrier_resets_hold(config, tmp_path):
    from llm_vil.engine import SaveStore
    from llm_vil.server.runtime import SimRuntime

    store = SaveStore(tmp_path / "b.sqlite")
    e = SimEngine(config, SlowDemo(0.1), store=store, clock=CognitionClock(t_max_seconds=5.0),
                  barrier_mode=True)
    for aid in e.order:
        e.world.apply(aid, AtomicAction(kind="wake"))
    rt = SimRuntime(e, paused=True)
    e.barrier_slice_s = 0.001
    while await rt.tick_once():
        pass
    assert rt._held
    meta = await rt.save_checkpoint("mid-barrier")
    assert meta and not rt._held and e._barrier_since is None and not e.jobs
    assert rt.health()["thinking_count"] == 0
    store.close()

"""Async ``SimEngine``: the only place where world/ and agent/ meet.

Per tick (deterministic actor order):
  1. ``World.tick_once`` (needs, movement, durative completions, daily growth, merchant)
  2. route change-only perceptions/events to agents
  3. wake / sleep / reflection triggers, conversation turns
  4. idle agents step their routine executor (``World.apply`` is the single gate) or get
     one cognition task (at most one in flight per agent, triggers are coalesced)
  5. settle cognition per the ``CognitionClock`` (dilate past ``T_max``) and adopt results
     with stale-view re-validation
  6. metrics, batched single-writer flush, pacing / sleep fast-forward
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

from ..agent.agent import Agent, RoutineDecision, TurnDecision, pending_offer
from ..agent.conversation import ConversationSession
from ..agent.memory import rumor_root
from ..llm.protocol import (
    Embedder,
    HashingEmbedder,
    LanguageModel,
    LLMRouter,
    Pricing,
    SharedRateGate,
    TokenBudget,
    dumps,
)
from ..schemas.agent import ConversationTurn, MemoryRecord, RoutineChoice
from ..schemas.config import VillageConfig
from ..schemas.protocol import (
    ClockDelta,
    ConversationDelta,
    Delta,
    GodEventDelta,
    ObjectChangeDelta,
    PathMoveDelta,
)
from ..schemas.world import ActionResult, AtomicAction, TilePos, Weather, WorldEvent
from ..world.world import World
from .clock import CognitionClock
from .save_store import SaveStore

TRIGGER_PRIORITY = ("plan_day", "reflect", "reflect_threshold", "weekly", "choose_routine")
SLEEP_TRIGGERS = frozenset({"reflect", "reflect_threshold", "weekly"})
INTERRUPTIBLE = frozenset({"idle", "walking", "resting", "wait"})
STUCK_LIMIT = 120          # sim minutes
STUCK_FALLBACK_AFTER = 45  # engine forces a safe wander after this much idleness
STARVING = 95.0
WATCHED_KINDS = frozenset({"monster", "shop_counter", "guild_counter"})
OBJECT_EVENTS = frozenset({"chopped", "tilled", "planted", "watered", "harvested", "foraged",
                           "mined", "gathered", "stored", "took", "chest_taken",
                           "order_fulfilled", "resource_spawned", "chest_missing_discovered",
                           "merchant_arrived", "merchant_left"})


@dataclass
class Job:
    kind: str
    task: asyncio.Task[Any]
    started: float
    reason: str = ""
    session_id: str | None = None


PHASE2_METRICS = {
    "p2p_trade": "p2p_trades", "shop_sale": "shop_sales", "shop_stocked": "shop_stocked",
    "crafted": "crafts", "mined": "ore_mined", "gathered": "herbs_gathered",
    "monster_defeated": "monsters_defeated", "fight_lost": "fights_lost", "fainted": "faints",
    "tool_repaired": "tools_repaired", "bounty_posted": "bounties_posted",
    "bounty_completed": "bounties_completed", "request_posted": "requests_posted",
    "drops_sold": "drops_sold",
}


def coin_gini(holdings: list[int]) -> float:
    """Gini coefficient of coin holdings (0 = equal, ->1 = one villager owns everything)."""
    xs = sorted(max(0, x) for x in holdings)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n


def _new_metrics() -> dict[str, int]:
    keys = (
        "wood_chopped", "crops_harvested", "harvest_events", "berries_foraged", "meals_cooked",
        "meals_eaten", "orders_fulfilled", "trades", "thefts", "gifts", "conversations_started",
        "conversations_completed", "conversations_declined", "conversation_turns",
        "hearsay_shared", "reflections", "weekly_syntheses", "archive_queries", "archive_hits",
        "archived_memories", "cognition_jobs", "stale_dropped", "cancelled_jobs", "rejections",
        "forced_replans", "stuck_incidents", "stuck_fallbacks", "max_stuck_minutes",
        "max_starving_minutes", "dilated_ticks", "god_events", "errors",
        "threshold_reflections", "checkpoints",
        # Phase 2 economy
        "p2p_trades", "shop_sales", "shop_stocked", "crafts", "crafts_hearth", "crafts_kiln",
        "crafts_bakery_oven", "crafts_forge_anvil", "crafts_cauldron", "ore_mined",
        "herbs_gathered", "monsters_defeated", "fights_lost", "faints", "tools_repaired",
        "bounties_posted", "bounties_completed", "requests_posted", "requests_fulfilled",
        "drops_sold", "coin_supply", "coin_supply_max", "coin_gini_milli",
    )
    return {k: 0 for k in keys}


class SimEngine:
    def __init__(self, config: VillageConfig, llm: LanguageModel, *,
                 store: SaveStore | None = None, clock: CognitionClock | None = None,
                 llm_timeout: float = 30.0, max_concurrency: int = 4,
                 budget: TokenBudget | None = None, perceive_every: int = 5,
                 checkpoint_every_days: int | None = None,
                 retry_base_delay_s: float = 0.5, embedder: Embedder | None = None,
                 json_repair_retries: int = 2,
                 pricing: Pricing | dict[str, Pricing] | None = None) -> None:
        self.config = config
        self.world = World(config)
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.router = LLMRouter(llm, gate=SharedRateGate(max_concurrency), budget=budget,
                                timeout=llm_timeout, ticks_per_day=config.ticks_per_day,
                                retries=json_repair_retries,
                                retry_base_delay_s=retry_base_delay_s, pricing=pricing)
        self.order = [v.id for v in config.villagers]
        self.agents = {v.id: Agent(v, config, self.router, self.embedder)
                       for v in config.villagers}
        self.clock = clock or CognitionClock(
            sim_minutes_per_real_second=config.clock.sim_minutes_per_real_second,
            t_max_seconds=0.0,
        )
        self.store = store
        self.perceive_every = perceive_every
        self.checkpoint_every_days = checkpoint_every_days
        self.jobs: dict[str, Job] = {}
        self.triggers: dict[str, dict[str, str]] = {aid: {} for aid in self.order}
        self.sessions: dict[str, ConversationSession] = {}
        self.actor_session: dict[str, str] = {}
        self.session_seq = 0
        self.metrics = _new_metrics()
        start = self.world.tick
        self.last_progress = {aid: start for aid in self.order}
        self.last_versions = {aid: 0 for aid in self.order}
        self._reset_watch()
        self.starving = {aid: 0 for aid in self.order}
        self.stuck_flagged = {aid: False for aid in self.order}
        self.pending_events: list[WorldEvent] = []
        self.pending_archive: list[tuple[MemoryRecord, int]] = []
        self.deltas: deque[Delta] = deque(maxlen=5000)
        self.errors: list[str] = []
        self._adopted: list[str] = []
        self._cancelled: list[asyncio.Task[Any]] = []  # awaited by quiesce(); never leaked
        self.checkpoint_pending = False

    # ================================================================ run
    @property
    def tick(self) -> int:
        return self.world.tick

    async def run_until(self, tick: int) -> None:
        while self.world.tick < tick:
            await self.step()

    async def run_days(self, days: float) -> None:
        await self.run_until(self.world.tick + int(days * self.config.ticks_per_day))

    async def step(self) -> None:
        w = self.world
        events = w.tick_once()
        self._record(events)
        per_agent = self._route(events)
        for aid in self.order:
            evs = per_agent[aid]
            for ev in evs:
                if ev.kind == "action_failed" and ev.actor_id == aid:
                    self.metrics["rejections"] += 1
                    if self.agents[aid].on_world_failure(ev.data.get("action", "?"),
                                                         ev.data.get("reason", "?")):
                        self.metrics["forced_replans"] += 1
                        self._trigger(aid, "choose_routine", "forced_replan")
                elif ev.kind == "fainted" and ev.actor_id == aid:
                    self.agents[aid].routine = None
                    self.agents[aid].progress = {}
                    self._trigger(aid, "choose_routine", "fainted")
            asleep = w.actors[aid].activity == "sleeping"
            if evs or (not asleep and w.tick % self.perceive_every == 0):
                self.agents[aid].observe(w.perceive(aid), evs)
            if self.agents[aid].wants_reflection() and aid not in self.jobs and not (
                {"reflect", "reflect_threshold"} & self.triggers[aid].keys()
            ):
                self._trigger(aid, "reflect_threshold", "importance")
        await self._embed_pending()
        if w.minute == 0:
            self._daily_maintenance()
        if w.minute % 60 == 0:
            self.deltas.append(ClockDelta(tick=w.tick, day=w.day, minute=w.minute,
                                          speed=self.clock.speed))
        self._wake_checks()
        self._progress_conversations()
        self._drive_agents()
        await self._settle()
        self._update_metrics()
        self._watch_objects()
        if self.checkpoint_pending and not self.jobs:  # clean tick boundary only
            self.checkpoint_pending = False
            self.checkpoint()
        await self.clock.pace(all(a.activity == "sleeping" for a in w.actors.values())
                              and not self.jobs)

    async def _embed_pending(self) -> None:
        """Embed new memories of every villager in one off-loop batch (blocking embedders).

        ONNX / HTTP embedders never run on the event loop: ``aembed`` does cache I/O here and
        model inference in a worker thread, so REST / WebSocket / LLM I/O keeps flowing.
        """
        if not getattr(self.embedder, "blocking", False):
            return
        batch: list[tuple[str, str, str]] = []
        for aid in self.order:
            batch.extend((aid, rid, text)
                         for rid, text in self.agents[aid].memory.pending_embeddings())
        if not batch:
            return
        vecs = await self.embedder.aembed([t for _, _, t in batch])  # type: ignore[attr-defined]
        per: dict[str, list[tuple[str, list[float]]]] = {}
        for (aid, rid, _), vec in zip(batch, vecs, strict=True):
            per.setdefault(aid, []).append((rid, vec))
        for aid, items in per.items():
            self.agents[aid].memory.apply_embeddings(items)

    async def quiesce(self) -> None:
        """Wait for every in-flight cognition and adopt it (for clean snapshots/shutdown)."""
        while self.jobs or self._cancelled:
            tasks = [j.task for j in self.jobs.values()] + self._cancelled
            await asyncio.wait(tasks)
            self._cancelled = [t for t in self._cancelled if not t.done()]
            self._harvest()

    # ================================================================ events
    def _record(self, events: list[WorldEvent]) -> None:
        self.pending_events.extend(events)
        for ev in events:
            k = ev.kind
            items = ev.data.get("items") or {}
            if k == "chopped":
                self.metrics["wood_chopped"] += items.get("wood", 0)
            elif k == "harvested":
                self.metrics["harvest_events"] += 1
                self.metrics["crops_harvested"] += items.get(ev.data.get("crop", ""), 0)
            elif k == "foraged":
                self.metrics["berries_foraged"] += items.get("berry", 0)
            elif k == "cooked":
                self.metrics["meals_cooked"] += 1
            elif k == "ate":
                self.metrics["meals_eaten"] += 1
            elif k == "order_fulfilled":
                self.metrics["orders_fulfilled"] += 1
                if ev.data.get("poster"):
                    self.metrics["requests_fulfilled"] += 1
            elif k == "traded":
                self.metrics["trades"] += 1
            elif k == "chest_taken":
                self.metrics["thefts"] += 1
            elif k == "gift":
                self.metrics["gifts"] += 1
            elif k in PHASE2_METRICS:
                self.metrics[PHASE2_METRICS[k]] += 1
                if k == "crafted":
                    key = f"crafts_{ev.data.get('station')}"
                    if key in self.metrics:
                        self.metrics[key] += 1
                elif k == "mined":
                    self.metrics["ore_mined"] += items.get("iron_ore", 0) - 1
            self._delta(ev)

    def _delta(self, ev: WorldEvent) -> None:
        if ev.kind == "path_move" and ev.actor_id:
            self.deltas.append(PathMoveDelta(
                actor_id=ev.actor_id, start_tick=ev.tick, step_ticks=ev.data["step_ticks"],
                waypoints=[TilePos(x=p[0], y=p[1]) for p in ev.data["waypoints"]],
                facing=ev.data["facing"],
            ))
        elif ev.kind in OBJECT_EVENTS:
            oid = ev.data.get("chest") or ev.target_id
            obj = self.world.objects.get(oid or "")
            if obj is not None:
                self.deltas.append(ObjectChangeDelta(object_id=obj.id, state=obj.state,
                                                     version=obj.version))

    def _deliver(self, events: list[WorldEvent]) -> None:
        """Route events that happen outside ``tick_once`` (apply results) to perception."""
        if not events:
            return
        tick = self.world.tick
        for aid, evs in self._route(events).items():
            if evs:
                self.agents[aid].observe_events(tick, evs)

    def _route(self, events: list[WorldEvent]) -> dict[str, list[WorldEvent]]:
        per: dict[str, list[WorldEvent]] = {aid: [] for aid in self.order}
        w = self.world
        for ev in events:
            if ev.kind in ("path_move", "action_started", "new_day"):
                continue
            for aid in self.order:
                if ev.actor_id == aid or aid in ev.witnesses:
                    per[aid].append(ev)
                elif ev.kind == "chest_taken" or ev.importance == 0:
                    continue
                elif ev.target_id == aid:
                    per[aid].append(ev)
                elif ev.pos is not None and w.can_perceive(aid, ev.pos.as_tuple()):
                    per[aid].append(ev)
        return per

    # ================================================================ triggers / jobs
    def _trigger(self, aid: str, kind: str, reason: str) -> None:
        self.triggers[aid].setdefault(kind, reason)  # coalesce: first reason wins

    def _pop_trigger(self, aid: str, asleep: bool) -> tuple[str, str] | None:
        trig = self.triggers[aid]
        for kind in TRIGGER_PRIORITY:
            if kind in trig and (not asleep or kind in SLEEP_TRIGGERS):
                return kind, trig.pop(kind)
        return None

    def _spawn(self, aid: str, kind: str, reason: str = "",
               session: ConversationSession | None = None,
               archive_hits: list[MemoryRecord] | None = None) -> None:
        agent = self.agents[aid]
        view = self.world.perceive(aid)
        if kind == "choose_routine":
            factory: Any = lambda: agent.decide_routine(view, reason)  # noqa: E731
        elif kind == "plan_day":
            factory = lambda: agent.decide_plan(view)  # noqa: E731
        elif kind in ("reflect", "reflect_threshold"):
            factory = lambda: agent.decide_reflection(view, reason)  # noqa: E731
        elif kind == "weekly":
            factory = lambda: agent.decide_weekly(view)  # noqa: E731
        elif kind == "turn":
            assert session is not None
            hits = archive_hits or []
            factory = lambda: agent.decide_turn(session, view, hits)  # noqa: E731
        else:
            raise ValueError(kind)
        task = asyncio.get_running_loop().create_task(self._guard(aid, kind, factory))
        self.jobs[aid] = Job(kind=kind, task=task, started=self.clock.time_fn(), reason=reason,
                             session_id=session.id if session else None)
        self.metrics["cognition_jobs"] += 1

    async def _guard(self, aid: str, kind: str, factory: Any) -> Any:
        """Coroutine is created lazily so a job cancelled before it starts leaks nothing."""
        try:
            return await factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # cognition must never crash the simulation
            self.metrics["errors"] += 1
            self.errors.append(f"{self.world.tick} {aid} {kind}: {exc!r}")
            return None

    def _cancel_job(self, aid: str, kinds: tuple[str, ...] = ("choose_routine", "plan_day"),
                    requeue: bool = True) -> str | None:
        """Cancel ``aid``'s in-flight cognition. The task is tracked until it really finishes
        so ``quiesce()`` awaits it. Only trigger kinds are ever re-queued."""
        job = self.jobs.get(aid)
        if job is None or job.kind not in kinds:
            return None
        job.task.cancel()
        self._cancelled.append(job.task)
        del self.jobs[aid]
        self.metrics["cancelled_jobs"] += 1
        if requeue and job.kind in TRIGGER_PRIORITY:
            self._trigger(aid, job.kind, f"{job.reason}+cancelled")
        return job.kind

    async def _settle(self) -> None:
        if self.jobs:
            tasks = [j.task for j in self.jobs.values()]
            if self.clock.must_wait([j.started for j in self.jobs.values()]):
                if self.clock.deterministic:
                    await asyncio.wait(tasks)
                else:
                    self.clock.dilated_ticks += 1
                    self.metrics["dilated_ticks"] += 1
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0)
        self._cancelled = [t for t in self._cancelled if not t.done()]
        self._harvest()

    def _harvest(self) -> None:
        self._adopted = []
        for aid in self.order:
            job = self.jobs.get(aid)
            if job is None or not job.task.done():
                continue
            del self.jobs[aid]
            if job.task.cancelled():
                continue
            self._adopt(aid, job, job.task.result())
        for aid in self._adopted:
            a = self.world.actors[aid]
            if a.action is None and a.activity == "idle" and aid not in self.actor_session:
                self._act(aid)

    def _adopt(self, aid: str, job: Job, result: Any) -> None:
        agent = self.agents[aid]
        a = self.world.actors[aid]
        if job.kind == "choose_routine":
            if a.activity == "sleeping" or aid in self.actor_session:
                return
            dec: RoutineDecision | None = result
            if dec is None:
                view = self.world.perceive(aid)
                dec = RoutineDecision(choice=agent._fallback_choice(view, agent.options(view)),
                                      view_version=view.world_version, fallback=True)
            for oid in dec.target_versions:  # resolved from the agent's known world
                cur = self.world.object_version(oid)
                if cur is None or cur > dec.view_version:  # mutated after the view was taken
                    self.metrics["stale_dropped"] += 1
                    self._trigger(aid, "choose_routine", f"stale_view:{oid}")
                    return
            agent.adopt_routine(dec)
            self._adopted.append(aid)
        elif job.kind == "plan_day":
            view = self.world.perceive(aid)
            agent.adopt_plan(result or agent._default_plan(), view)
            self._trigger(aid, "choose_routine", "plan_ready")
        elif job.kind in ("reflect", "reflect_threshold"):
            mode = "importance" if job.kind == "reflect_threshold" else "end_of_day"
            if agent.adopt_reflection(result, self.world.perceive(aid), mode):
                self.metrics["reflections"] += 1
                if mode == "importance":
                    self.metrics["threshold_reflections"] += 1
        elif job.kind == "weekly":
            if agent.adopt_weekly(result, self.world.perceive(aid)):
                self.metrics["weekly_syntheses"] += 1
        elif job.kind == "turn":
            self._apply_turn(job.session_id or "", aid, result)

    # ================================================================ agents
    def _wake_checks(self) -> None:
        w = self.world
        minute = w.minute
        for aid in self.order:
            a = w.actors[aid]
            spec = self.agents[aid].spec
            if a.activity == "sleeping" and spec.wake_minute <= minute < spec.sleep_minute:
                if aid in self.jobs:  # finish nightly reflection first
                    continue
                res = w.apply(aid, AtomicAction(kind="wake"))
                self._record(res.events)
                agent = self.agents[aid]
                agent.finish_routine()
                if agent.plan_day_index != w.day:
                    self._trigger(aid, "plan_day", "wake")
                self._trigger(aid, "choose_routine", "wake")

    def _drive_agents(self) -> None:
        w = self.world
        for aid in self.order:
            if aid in self.jobs or aid in self.actor_session:
                continue
            a = w.actors[aid]
            asleep = a.activity == "sleeping"
            trig = self._pop_trigger(aid, asleep)
            if trig is not None:
                kind, reason = trig
                if kind == "choose_routine" and a.action is not None:
                    self._trigger(aid, kind, reason)  # decide when the durative action ends
                    continue
                if kind == "reflect" and self.agents[aid].last_reflection_day == w.day:
                    continue
                self._spawn(aid, kind, reason)
                continue
            if asleep or a.action is not None:
                continue
            if self.agents[aid].routine is None:
                self._spawn(aid, "choose_routine", "idle")
                continue
            self._act(aid)

    def _act(self, aid: str) -> None:
        w = self.world
        agent = self.agents[aid]
        for _ in range(4):  # follow instant / no-op actions within the same tick
            view = w.perceive(aid)
            act = agent.next_action(view)
            if act is None:
                self._trigger(aid, "choose_routine", "routine_done")
                return
            if act.kind == "talk":
                self._invite(aid, act)
                return
            res = w.apply(aid, act)
            self._record(res.events)
            self._deliver(res.events)
            forced = agent.on_apply_result(res)
            if not res.ok:
                self.metrics["rejections"] += 1
                if forced:
                    self.metrics["forced_replans"] += 1
                self._trigger(aid, "choose_routine", "forced_replan" if forced else
                              f"rejected:{res.reason}")
                return
            if act.kind == "sleep":
                self._on_sleep(aid)
                return
            if w.actors[aid].action is not None:
                return

    def _on_sleep(self, aid: str) -> None:
        agent = self.agents[aid]
        agent.finish_routine()
        day = self.world.day
        if agent.last_reflection_day != day:
            self._trigger(aid, "reflect", "end_of_day")
        if agent.wants_weekly(day):  # robust to sleeping past midnight
            self._trigger(aid, "weekly", "end_of_week")

    def _economy_gauges(self) -> None:
        holdings = self.world.coin_holdings()
        villagers = [holdings.get(aid, 0) for aid in self.order]
        supply = sum(holdings.values())
        self.metrics["coin_supply"] = supply
        self.metrics["coin_supply_max"] = max(self.metrics["coin_supply_max"], supply)
        self.metrics["coin_gini_milli"] = round(coin_gini(villagers) * 1000)

    def _daily_maintenance(self) -> None:
        self._economy_gauges()
        tick = self.world.tick
        for aid in self.order:
            archived = self.agents[aid].daily_maintenance(tick)
            self.pending_archive.extend((rec, tick) for rec in archived)
            self.metrics["archived_memories"] += len(archived)
        self.flush()
        n = self.checkpoint_every_days
        if n and self.store is not None and self.world.day % n == 0:
            self.checkpoint_pending = True  # written at the end of a quiescent step

    def _watch_objects(self) -> None:
        """Stream wandering monsters and shop shelves to viewers (UI only, not sim state)."""
        for oid in self.watched:
            o = self.world.objects[oid]
            if o.version == self.watched[oid]:
                continue
            self.watched[oid] = o.version
            view = self.world._obj_view(o)
            self.deltas.append(ObjectChangeDelta(
                object_id=oid, state=o.state, version=o.version, data=view.data,
                pos=TilePos(x=o.pos[0], y=o.pos[1]),
            ))

    def _reset_watch(self) -> None:
        self.watched = {o.id: o.version for o in self.world.objects.values()
                        if o.kind in WATCHED_KINDS}

    def _update_metrics(self) -> None:
        w = self.world
        tick = w.tick
        for aid in self.order:
            a = w.actors[aid]
            busy = a.activity in ("sleeping", "in_conversation") or a.action is not None
            if a.version != self.last_versions[aid] or busy:
                self.last_progress[aid] = tick
                self.stuck_flagged[aid] = False
            self.last_versions[aid] = a.version
            idle = tick - self.last_progress[aid]
            self.metrics["max_stuck_minutes"] = max(self.metrics["max_stuck_minutes"], idle)
            if idle > STUCK_LIMIT and not self.stuck_flagged[aid]:
                self.stuck_flagged[aid] = True
                self.metrics["stuck_incidents"] += 1
            if idle >= STUCK_FALLBACK_AFTER and idle % STUCK_FALLBACK_AFTER == 0 and (
                aid not in self.jobs and aid not in self.actor_session
            ):
                self.metrics["stuck_fallbacks"] += 1
                self.agents[aid].adopt_routine(RoutineDecision(
                    choice=RoutineChoice(routine="wander", target="plaza", reason="stuck"),
                    view_version=w.world_version, fallback=True,
                ))
            if a.hunger >= STARVING:
                self.starving[aid] += 1
                self.metrics["max_starving_minutes"] = max(self.metrics["max_starving_minutes"],
                                                           self.starving[aid])
            else:
                self.starving[aid] = 0

    # ================================================================ conversations
    def _invite(self, aid: str, act: AtomicAction) -> None:
        w = self.world
        target = act.target_id or ""
        self.session_seq += 1
        session = ConversationSession(
            id=f"conv_{self.session_seq:05d}", initiator_id=aid, target_id=target,
            topic=act.item or "안부", start_tick=w.tick,
            max_turns=self.config.conversation.max_turns,
        )
        t = w.actors.get(target)
        me = w.actors[aid]
        available = (
            t is not None and target not in self.actor_session and t.activity in INTERRUPTIBLE
            and max(abs(t.pos[0] - me.pos[0]), abs(t.pos[1] - me.pos[1])) <= 3
        )
        if not available or not self.agents[target].consider_invite(aid, w.perceive(target)):
            session.decline(w.tick)
            self.metrics["conversations_declined"] += 1
            self.agents[aid].apply_conversation_outcome(session.outcome(), w.tick)
            self._trigger(aid, "choose_routine", "invite_declined")
            return
        session.accept(w.tick)
        self._cancel_job(target)
        for pid in (aid, target):
            self._record(w.interrupt(pid, "conversation"))
            w.set_activity(pid, "in_conversation")
            self.actor_session[pid] = session.id
        w.face_toward(aid, w.actors[target].pos)
        w.face_toward(target, w.actors[aid].pos)
        self.sessions[session.id] = session
        self.metrics["conversations_started"] += 1
        self.deltas.append(ConversationDelta(session_id=session.id,
                                             participants=session.participants, state="active"))

    def _progress_conversations(self) -> None:
        tick = self.world.tick
        for sid in sorted(self.sessions):
            s = self.sessions[sid]
            if s.state != "active" or tick < s.next_turn_tick:
                continue
            speaker = s.next_speaker
            if speaker in self.jobs:
                continue
            partner = s.partner_of(speaker)
            hits: list[MemoryRecord] = []
            if self.store is not None and len(s.turns) < 2:
                hits = self.store.search_archive(speaker, subject=partner, query=s.topic)
                self.metrics["archive_queries"] += 1
                self.metrics["archive_hits"] += len(hits)
            self._spawn(speaker, "turn", session=s, archive_hits=hits)

    def _apply_turn(self, sid: str, aid: str, decision: TurnDecision | None) -> None:
        w = self.world
        s = self.sessions.get(sid)
        if s is None or s.state != "active":
            return
        partner = s.partner_of(aid)
        if decision is None:
            decision = TurnDecision(turn=ConversationTurn(
                speaker_id=aid, listener_id=partner, tick=w.tick,
                utterance="미안, 잠깐 다른 생각을 했어.", end_conversation=True), fallback=True)
        turn = decision.turn
        if turn.gift_item != "none":
            res = w.apply(aid, AtomicAction(kind="give_item", target_id=partner,
                                            item=turn.gift_item, qty=1))
            self._record(res.events)
            if not res.ok:
                turn = turn.model_copy(update={"gift_item": "none"})
        offer = pending_offer(s.turns, partner)
        if turn.trade_response == "accept":
            ok = False
            if offer is not None:
                ok, _why, evs = w.swap_items(partner, aid, offer["give_item"], offer["give_qty"],
                                             offer["want_item"], offer["want_qty"])
                self._record(evs)
                self._deliver(evs)
            if not ok:
                turn = turn.model_copy(update={"trade_response": "decline"})
        s.add_turn(turn, self.config.conversation.ticks_per_turn)
        self.metrics["conversation_turns"] += 1
        pos = w.actors[aid].pos
        ev = WorldEvent(tick=w.tick, kind="utterance", actor_id=aid, target_id=partner,
                        pos=TilePos(x=pos[0], y=pos[1]), importance=2,
                        data={"text": turn.utterance, "to": [partner], "session": sid})
        self.pending_events.append(ev)
        for bid in self.order:
            if bid not in (aid, partner) and w.can_perceive(bid, pos):
                self.agents[bid].observe(w.perceive(bid), [ev])
        self.deltas.append(ConversationDelta(
            session_id=sid, participants=s.participants, state=s.state, speaker_id=aid,
            utterance=turn.utterance, emotion=turn.emotion,
        ))
        if decision.shared is not None:
            listener = self.agents[partner]
            self.agents[aid].mark_told(partner, decision.shared)
            if not listener.memory.knows_rumor(rumor_root(decision.shared)):  # no echo loops
                listener.receive_hearsay(decision.shared, aid, w.tick)
                self.metrics["hearsay_shared"] += 1
        if s.state == "ended":
            self._end_session(s)

    def _end_session(self, s: ConversationSession) -> None:
        w = self.world
        outcome = s.outcome()
        for pid in s.participants:
            self.agents[pid].apply_conversation_outcome(outcome, w.tick)
            if w.actors[pid].activity == "in_conversation":
                w.set_activity(pid, "idle")
            partner = s.partner_of(pid)
            w.adjust_mood(pid, outcome.mean_valence.get(partner, 0.0) * 5)
            self.actor_session.pop(pid, None)
        self.sessions.pop(s.id, None)
        if len(outcome.turns) >= 2:
            self.metrics["conversations_completed"] += 1
        self.pending_events.append(WorldEvent(
            tick=w.tick, kind="conversation_ended", actor_id=s.initiator_id, target_id=s.target_id,
            importance=1, data={"session": s.id, "reason": outcome.end_reason,
                                "turns": len(outcome.turns), "topic": s.topic},
        ))
        self.deltas.append(ConversationDelta(session_id=s.id, participants=s.participants,
                                             state="ended"))

    def _interrupt_sessions(self, reason: str) -> None:
        for sid in sorted(self.sessions):
            s = self.sessions[sid]
            for pid in s.participants:
                self._cancel_job(pid, ("turn",), requeue=False)
            s.interrupt(self.world.tick, "interrupt")
            self._end_session(s)

    # ================================================================ God mode
    def _god(self, kind: str, events: list[WorldEvent], **data: Any) -> None:
        """Log a God command (SQLite event log + delta) and make its world events village-wide
        perceptions for every villager."""
        w = self.world
        self.pending_events.append(WorldEvent(tick=w.tick, kind=f"god_{kind}", importance=9,
                                              data=data))
        self._record(events)
        for aid in self.order:
            if events:
                self.agents[aid].observe_events(w.tick, events)
        self.metrics["god_events"] += 1
        self.deltas.append(GodEventDelta(kind=kind, data=data))

    def set_weather(self, weather: Weather | str) -> None:
        evs = self.world.set_weather(Weather(weather))
        self._god("set_weather", evs, weather=Weather(weather).value)

    def spawn_resource(self, kind: str, pos: tuple[int, int], state: str | None = None) -> None:
        evs = self.world.spawn_resource(kind, pos, state)
        self._god("spawn_resource", evs, object_kind=kind, pos=list(pos))

    def inject_event(self, text: str, importance: int = 9) -> None:
        """Village event: enters every memory, interrupts sessions & in-flight intents."""
        self._god("inject_event", [], text=text)
        self._interrupt_sessions("external_event")
        self._broadcast(text, importance)
        for aid in self.order:
            if self._cancel_job(aid, requeue=False) == "plan_day":
                self._trigger(aid, "plan_day", "external_event")
            if self.world.actors[aid].activity != "sleeping":
                self._record(self.world.interrupt(aid, "external_event"))
                self.agents[aid].finish_routine()
                self.triggers[aid].pop("choose_routine", None)
                self._trigger(aid, "choose_routine", "external_event")

    def inject_rumor(self, target_id: str, about_id: str, text: str,
                     confidence: float = 0.7, valence: float = -0.5) -> None:
        self._god("inject_rumor", [], target=target_id, about=about_id, text=text)
        agent = self.agents[target_id]
        agent.memory.add(self.world.tick, "hearsay", f"소문: {text}", 6, valence=valence,
                         subjects=[about_id], source_actor_id="rumor", confidence=confidence)
        if valence <= -0.3 and about_id in agent.relationships.rels:
            agent.relationships.apply_event(about_id, "hearsay_theft", scale=confidence,
                                            note=f"{self.world.tick}: 소문")

    def _broadcast(self, text: str, importance: int) -> None:
        for aid in self.order:
            self.agents[aid].memory.add(self.world.tick, "external_event", text, importance,
                                        subjects=["village"], source_actor_id="god")

    # ================================================================ persistence
    def flush(self) -> None:
        tape = self.router.drain_tape()
        events, self.pending_events = self.pending_events, []
        archive, self.pending_archive = self.pending_archive, []
        if self.store is not None:
            self.store.write_batch(events=events, archive=archive, tape=tape)

    def snapshot(self) -> dict[str, Any]:
        if self.jobs:
            raise RuntimeError("cannot snapshot while cognition is in flight; quiesce() first")
        return {
            "tick": self.world.tick,
            "world": self.world.snapshot(),
            "agents": {aid: self.agents[aid].snapshot() for aid in self.order},
            "sessions": {sid: s.snapshot() for sid, s in sorted(self.sessions.items())},
            "actor_session": dict(sorted(self.actor_session.items())),
            "session_seq": self.session_seq,
            "triggers": {aid: dict(t) for aid, t in self.triggers.items()},
            "metrics": dict(self.metrics),
            "last_progress": dict(self.last_progress),
            "last_versions": dict(self.last_versions),
            "starving": dict(self.starving),
            "stuck_flagged": dict(self.stuck_flagged),
            "checkpoint_pending": self.checkpoint_pending,
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        if self.jobs or any(not t.done() for t in self._cancelled):
            raise RuntimeError("cannot load a snapshot while cognition is in flight")
        self._cancelled = []
        self.pending_events = []
        self.pending_archive = []
        self.deltas.clear()
        self.errors = []
        self.checkpoint_pending = bool(data.get("checkpoint_pending", False))
        self.world = World.restore(self.config, data["world"])
        for aid in self.order:
            self.agents[aid].restore(data["agents"][aid])
        self.sessions = {sid: ConversationSession.model_validate(s)
                         for sid, s in data["sessions"].items()}
        self.actor_session = dict(data["actor_session"])
        self.session_seq = data["session_seq"]
        self.triggers = {aid: dict(t) for aid, t in data["triggers"].items()}
        self.metrics = {**_new_metrics(), **data["metrics"]}
        self.last_progress = dict(data["last_progress"])
        self.last_versions = dict(data["last_versions"])
        self._reset_watch()
        self.starving = dict(data["starving"])
        self.stuck_flagged = dict(data["stuck_flagged"])

    def checkpoint(self, label: str = "") -> int:
        if self.store is None:
            raise RuntimeError("no SaveStore attached")
        self.flush()
        self.metrics["checkpoints"] += 1
        return self.store.save_checkpoint(self.world.tick, dumps(self.snapshot()), label)

    def fresh_snapshot(self) -> dict[str, Any]:
        """Snapshot of a brand-new village with this engine's config (for reset)."""
        return SimEngine(self.config, self.router.llm, embedder=self.embedder).snapshot()

    @classmethod
    def from_checkpoint(cls, config: VillageConfig, llm: LanguageModel, store: SaveStore,
                        at_or_before: int | None = None, **kwargs: Any) -> SimEngine:
        tick, data = store.load_checkpoint(at_or_before)
        store.truncate_after(tick)
        engine = cls(config, llm, store=store, **kwargs)
        engine.load_snapshot(data)
        return engine

    # ================================================================ introspection
    def summary(self) -> dict[str, Any]:
        rel_moves = 0.0
        for agent in self.agents.values():
            for r in agent.relationships.rels.values():
                rel_moves += abs(r.familiarity) + abs(r.affection) + abs(r.trust) + abs(r.respect)
        return {
            "tick": self.world.tick, "day": self.world.day, "metrics": dict(self.metrics),
            "world": self.world.stats(), "router": self.router.stats.model_dump(),
            "relationship_magnitude": round(rel_moves, 4),
            "clock": {"dilated_ticks": self.clock.dilated_ticks,
                      "fast_forward_ticks": self.clock.fast_forward_ticks},
            "errors": list(self.errors),
        }

    def drain_deltas(self) -> list[Delta]:
        out = list(self.deltas)
        self.deltas.clear()
        return out

    def submit_intent(self, aid: str, action: AtomicAction) -> ActionResult:
        """Direct atomic intent through the single gate (Phase 2 human ``Actor`` boundary)."""
        res = self.world.apply(aid, action)
        self._record(res.events)
        return res

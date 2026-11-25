"""``SimRuntime``: drives the SimEngine in the server's event loop and fans out deltas.

The engine is single-threaded asyncio; REST handlers and WebSocket commands run on the same
loop, so they read/mutate engine state between ticks without locks.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import Any

from ..agent.memory import rumor_text
from ..engine import CognitionClock, SimEngine
from ..engine.sim_engine import coin_gini
from ..schemas.base import Frozen
from ..schemas.world import Weather, WorldEvent
from .replay import DEFAULT_CAP, SPEECH_TICKS, ReplayRecorder

ALLOWED_SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_CLIENT_BACKLOG = 200
FLUSH_INTERVAL_S = 0.05
AGENT_STATE_EVERY = 10  # ticks; plus whenever activity changes
TIMELINE_SKIP = frozenset({"path_move", "action_started", "action_completed", "new_day",
                           "order_posted", "rested", "waited"})
SPAWN_KINDS = {"wood": ("tree", "mature"), "berry_bush": ("berry_bush", "ripe"),
               "seeds": ("chest", None)}


class AgentStateDelta(Frozen):
    type: str = "agent_state"
    actor_id: str
    x: int
    y: int
    facing: int
    activity: str
    hunger: float
    stamina: float
    mood: float
    hp: float = 100.0
    fainted: bool = False
    routine: str | None = None
    in_conversation: bool = False


def event_payload(ev: WorldEvent) -> dict[str, Any]:
    d = ev.model_dump(mode="json")
    d["type"] = "event"
    d["data"] = {k: v for k, v in d["data"].items() if k not in ("missing",)}
    if ev.kind == "chest_taken":  # thief identity is only known to witnesses
        d["actor_id"] = None
    return d


MANUAL_LABEL = "수동 저장"  # a user's unnamed save must never be pruned as an automatic one
MAX_RUN_DAYS = 3650
UNSET: Any = object()


def normalize_max_days(days: Any) -> int | None:
    """``None`` / 0 = unlimited; anything else must be a whole number of days."""
    if days is None:
        return None
    if isinstance(days, bool) or not 0 <= float(days) <= MAX_RUN_DAYS or float(days) % 1:
        raise ValueError(f"days must be an integer between 0 and {MAX_RUN_DAYS}")
    return int(days) or None


class SimRuntime:
    def __init__(self, engine: SimEngine, *, speed: float = 1.0, paused: bool = True,
                 replay_cap: int = DEFAULT_CAP, max_run_days: int | None = UNSET) -> None:
        self.engine = engine
        self.speed = speed
        self.paused = paused
        # run window: auto-pause after this many sim-days so nothing runs unattended forever
        self.max_run_days = normalize_max_days(
            engine.config.clock.max_run_days if max_run_days is UNSET else max_run_days)
        self.limit_reached = False
        self._window_start = engine.world.tick
        self.replay = ReplayRecorder(engine.store, replay_cap)
        self._tick_events: list[dict[str, Any]] = []
        self._speech: dict[str, tuple[int, str]] = {}
        self._fainted_until: dict[str, int] = {}
        self._held = False  # last step was held at the decision barrier
        self._last_thinking: list[str] = []
        self.clients: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self._lock = asyncio.Lock()  # serializes engine.step() with client commands
        self.dropped_clients = 0
        self.timeline: deque[dict[str, Any]] = deque(maxlen=300)
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._step_budget = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_activity: dict[str, str] = {}
        self._captured: list[WorldEvent] = []
        self._seen = 0
        # Pacing lives here, outside the engine lock: the engine runs unpaced (it keeps its
        # cognition-aware dilation), and manual steps while paused are never throttled.
        self.pacer = CognitionClock(
            speed=speed,
            sim_minutes_per_real_second=engine.config.clock.sim_minutes_per_real_second,
        )
        engine.clock.set_speed(None)
        orig_flush = engine.flush

        def capturing_flush() -> None:  # keep events the engine flushes mid-step
            self._capture()
            self._seen = 0
            orig_flush()

        engine.flush = capturing_flush  # type: ignore[method-assign]

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.engine.quiesce()
        self.replay.flush()

    async def _run(self) -> None:
        while True:
            if self.paused and self._step_budget <= 0:
                self._flush(force=True)
                self._wake.clear()
                await self._wake.wait()
                continue
            manual = self.paused and self._step_budget > 0
            if not await self.tick_once():
                await asyncio.sleep(0)  # held at the decision barrier: no pacing, no budget
                continue
            if manual:
                self._step_budget = max(0, self._step_budget - 1)
                await asyncio.sleep(0)  # never starve REST / WebSocket handlers
            else:
                w = self.engine.world
                all_asleep = all(a.activity == "sleeping" for a in w.actors.values())
                await self.pacer.pace(all_asleep and not self.engine.jobs)
                await asyncio.sleep(0)

    async def tick_once(self) -> bool:
        """One engine step under the lock; False when the decision barrier held the tick."""
        async with self._lock:
            return await self._tick_locked()

    def _clock_msg(self) -> dict[str, Any]:
        w = self.engine.world
        thinking = self.engine.thinking_actors
        return {"type": "clock", "tick": w.tick, "day": w.day, "minute": w.minute,
                "speed": self.speed, "paused": self.paused, "weather": w.weather.value,
                "merchant_present": w.merchant_present,
                "thinking_count": len(thinking), "thinking_actors": thinking,
                **self._limit_fields()}

    # ------------------------------------------------------------ run window (max days)
    def _limit_fields(self) -> dict[str, Any]:
        return {"max_run_days": self.max_run_days,
                "run_days_elapsed": round(self._run_days(), 3),
                "limit_reached": self.limit_reached}

    def _run_days(self) -> float:
        e = self.engine
        return (e.world.tick - self._window_start) / e.config.ticks_per_day

    def _new_window(self) -> None:
        """(lock held) start counting days from now."""
        self._window_start = self.engine.world.tick
        self.limit_reached = False

    def _check_limit(self) -> None:
        """(lock held) pause exactly when the run window is used up."""
        days = self.max_run_days
        if days is None or self.limit_reached or self._run_days() < days:
            return
        self.limit_reached = True
        self.paused = True
        self._step_budget = 0
        w = self.engine.world
        notice = {"type": "event", "tick": w.tick, "kind": "run_limit_reached",
                  "actor_id": None, "target_id": None, "pos": None, "importance": 5,
                  "witnesses": [], "data": {
                      "days": days,
                      "text": f"최대 실행 일수({days}일)에 도달해 자동 일시정지되었습니다"}}
        self.timeline.append(notice)
        self._emit(notice)
        self._emit(self._clock_msg())

    def _reset_hold(self) -> None:
        """(lock held) jobs were finished / cancelled outside a step: forget the held state
        and tell clients the current thinking status right away."""
        self.engine.reset_barrier()
        self._held = False
        self._last_thinking = list(self.engine.thinking_actors)
        self._emit(self._clock_msg())

    async def _tick_locked(self) -> bool:
        e = self.engine
        held = self._held
        if not await e.step():
            thinking = e.thinking_actors
            if not held or thinking != self._last_thinking:  # status only on change
                self._last_thinking = thinking
                self._emit(self._clock_msg())
                self._flush(force=True)
            self._held = True
            return False
        self._held = False
        self._last_thinking = []
        self._capture()
        w = e.world
        self._emit(self._clock_msg())
        for d in e.drain_deltas():
            payload = d.model_dump(mode="json")
            if payload["type"] == "object_change":
                obj = w.objects.get(payload["object_id"])
                if obj is not None:
                    payload["data"] = w._obj_view(obj).data
                    if obj.kind == "monster":
                        payload["pos"] = {"x": obj.pos[0], "y": obj.pos[1]}
            if payload["type"] == "clock":
                continue  # replaced by the per-tick clock message above
            if payload["type"] == "conversation" and payload.get("utterance"):
                self._speech[payload["speaker_id"]] = (w.tick, payload["utterance"])
            if payload["type"] == "path_move":  # paths start from the actor's tile this tick
                a = w.actors[payload["actor_id"]]
                payload["start_pos"] = {"x": a.pos[0], "y": a.pos[1]}
            self._emit(payload)
        for aid in e.order:
            a = w.actors[aid]
            if a.activity != self._last_activity.get(aid) or w.tick % AGENT_STATE_EVERY == 0:
                self._last_activity[aid] = a.activity
                self._emit(self.agent_state(aid).model_dump(mode="json"))
        self._record_frame()
        self._check_limit()
        # barrier release / auto-pause: the batch leaves at once
        self._flush(force=held or self.limit_reached)
        return True

    # ------------------------------------------------------------ replay
    def _record_frame(self) -> None:
        """Compact visual frame of this tick (renderer state only, no minds)."""
        e = self.engine
        w = e.world
        actors = []
        for aid in e.order:
            a = w.actors[aid]
            agent = e.agents[aid]
            said = self._speech.get(aid)
            speech = said[1] if said and 0 <= w.tick - said[0] < SPEECH_TICKS else None
            actors.append([aid, a.pos[0], a.pos[1], a.facing, a.activity,
                           agent.routine.routine if agent.routine else None,
                           round(a.hunger, 1), round(a.stamina, 1), round(a.mood, 1),
                           aid in e.actor_session, speech, round(a.hp, 1), self._is_fainted(aid)])
        objects = []
        for o in w.objects.values():
            data = {k: v for k, v in o.data.items()
                    if k in ("crop", "growth", "species", "stock", "prices", "open", "treasury")}
            if o.kind == "monster":
                objects.append([o.id, o.state, data, [o.pos[0], o.pos[1]]])
            elif data:
                objects.append([o.id, o.state, data])
            else:
                objects.append([o.id, o.state])
        frame = {"w": w.weather.value, "mp": w.merchant_present, "a": actors, "o": objects,
                 "ev": self._tick_events}
        self._tick_events = []
        self.replay.record(w.tick, w.day, w.minute, frame)

    def replay_range(self, include_ticks: bool = False) -> dict[str, Any]:
        ticks = self.replay.ticks()
        tpd = self.engine.config.ticks_per_day
        if not ticks:
            return {"min_tick": None, "max_tick": None, "count": 0,
                    **({"ticks": []} if include_ticks else {}),
                    "live_tick": self.engine.world.tick}
        lo, hi = ticks[0], ticks[-1]
        return {"min_tick": lo, "max_tick": hi, "count": len(ticks),
                **({"ticks": ticks} if include_ticks else {}),
                "min": {"day": lo // tpd, "minute": lo % tpd},
                "max": {"day": hi // tpd, "minute": hi % tpd},
                "live_tick": self.engine.world.tick}

    def replay_frame(self, tick: int) -> dict[str, Any] | None:
        """The recorded frame at (or nearest before) ``tick``, shaped like a live snapshot
        so the client store renders it identically. Static data comes from the live world."""
        hit = self.replay.frame_at(tick)
        if hit is None:
            return None
        at, frame = hit
        e = self.engine
        w = e.world
        cfg = e.config
        snap = self.snapshot()  # map / relationships / static object + villager info
        objects = []
        for row in frame["o"]:
            obj = w.objects.get(row[0])
            if obj is None:
                continue
            view = w._obj_view(obj).model_dump(mode="json")
            view["state"] = row[1]
            view["data"] = {
                **{k: v for k, v in view["data"].items()
                   if k not in ("crop", "growth", "species", "stock", "prices", "open",
                                "treasury")},
                **(row[2] if len(row) > 2 else {}),
            }
            if len(row) > 3 and row[3]:
                view["pos"] = {"x": row[3][0], "y": row[3][1]}
            objects.append(view)
        info = {a["actor_id"]: a for a in snap["actors"]}
        actors = []
        for arow in frame["a"]:
            aid, x, y, facing, activity, routine, hunger, stamina, mood, conv, speech = arow[:11]
            hp = arow[11] if len(arow) > 11 else 100.0
            fainted = bool(arow[12]) if len(arow) > 12 else False
            base = info.get(aid)
            if base is None:
                continue
            actors.append({**base, "x": x, "y": y, "facing": facing, "activity": activity,
                           "routine": routine, "hunger": hunger, "stamina": stamina,
                           "mood": mood, "hp": hp, "fainted": fainted,
                           "in_conversation": conv, "path": None, "speech": speech})
        snap.update({
            "replay": True,
            "clock": {**snap["clock"], "tick": at, "day": at // cfg.ticks_per_day,
                      "minute": at % cfg.ticks_per_day, "paused": True, "weather": frame["w"],
                      "merchant_present": frame["mp"]},
            "objects": objects, "actors": actors, "events": self.replay.events_upto(at),
            "orders": [], "sessions": [],
        })
        return snap

    def _capture(self) -> None:
        pending = self.engine.pending_events
        new = pending[self._seen:]
        self._seen = len(pending)
        for ev in new:
            if ev.kind == "fainted" and ev.actor_id:
                self._fainted_until[ev.actor_id] = ev.tick + 15
            if ev.kind in TIMELINE_SKIP or (ev.importance < 1 and ev.kind != "utterance"):
                continue
            payload = event_payload(ev)
            self.timeline.append(payload)
            self._tick_events.append(payload)
            self._emit(payload)

    def _is_fainted(self, aid: str) -> bool:
        w = self.engine.world
        a = w.actors[aid]
        return a.hp < self.engine.config.combat.faint_hp or w.tick < self._fainted_until.get(aid, 0)

    # ------------------------------------------------------------ fan-out
    def _emit(self, msg: dict[str, Any]) -> None:
        self._buffer.append(msg)

    def _flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not self._buffer or (not force and now - self._last_flush < FLUSH_INTERVAL_S
                                and len(self._buffer) < 400):
            return
        batch = {"type": "batch", "tick": self.engine.tick, "deltas": self._buffer}
        self._buffer = []
        self._last_flush = now
        for q in list(self.clients):
            if q.qsize() > MAX_CLIENT_BACKLOG:
                # slow client: tell its socket to close (it auto-reconnects and resyncs
                # from a fresh snapshot) instead of buffering without bound
                self.clients.discard(q)
                self.dropped_clients += 1
                q.put_nowait(None)
                continue
            q.put_nowait(batch)

    def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any] | None]) -> None:
        self.clients.discard(q)

    # ------------------------------------------------------------ views
    def agent_state(self, aid: str) -> AgentStateDelta:
        a = self.engine.world.actors[aid]
        agent = self.engine.agents[aid]
        return AgentStateDelta(
            actor_id=aid, x=a.pos[0], y=a.pos[1], facing=a.facing, activity=a.activity,
            hunger=round(a.hunger, 2), stamina=round(a.stamina, 2), mood=round(a.mood, 2),
            hp=round(a.hp, 2), fainted=self._is_fainted(aid),
            routine=agent.routine.routine if agent.routine else None,
            in_conversation=aid in self.engine.actor_session,
        )

    def _metrics_view(self) -> dict[str, Any]:
        e = self.engine
        m: dict[str, Any] = dict(e.metrics)
        holdings = e.world.coin_holdings()
        villagers = [holdings.get(aid, 0) for aid in e.order]
        m["coin_supply"] = sum(holdings.values())
        m["coin_gini"] = round(coin_gini(villagers), 4)
        return m

    def health(self) -> dict[str, Any]:
        w = self.engine.world
        thinking = self.engine.thinking_actors
        return {"ok": True, "day": w.day, "minute": w.minute, "tick": w.tick,
                "speed": self.speed, "paused": self.paused, "actors_count": len(w.actors),
                "thinking_count": len(thinking), "thinking_actors": thinking,
                "barrier_mode": self.engine.barrier_active, **self._limit_fields()}

    def snapshot(self) -> dict[str, Any]:
        e = self.engine
        w = e.world
        cfg = e.config
        actors = []
        for aid in e.order:
            a = w.actors[aid]
            spec = e.agents[aid].spec
            path = None
            if a.action is not None and a.action.action.kind == "move_to" and a.action.path:
                path = {"start_tick": a.action.next_step_tick - a.action.step_ticks,
                        "step_ticks": a.action.step_ticks,
                        "waypoints": [{"x": p[0], "y": p[1]} for p in a.action.path]}
            actors.append({
                **self.agent_state(aid).model_dump(mode="json"), "name": spec.name,
                "occupation": spec.occupation, "profession": spec.profession,
                "skills": list(spec.skills), "home_area": spec.home_area,
                "equipment": {k: dict(v) for k, v in a.equipment.items()},
                "path": path,
            })
        return {
            "type": "snapshot",
            "map": {"width": cfg.width, "height": cfg.height, "name": cfg.name,
                    "water": [list(r) for r in cfg.water],
                    "areas": [a.model_dump(mode="json") for a in cfg.areas]},
            "clock": {"tick": w.tick, "day": w.day, "minute": w.minute, "speed": self.speed,
                      "paused": self.paused, "weather": w.weather.value,
                      "merchant_present": w.merchant_present,
                      "ticks_per_day": cfg.ticks_per_day,
                      "sim_minutes_per_real_second": cfg.clock.sim_minutes_per_real_second,
                      "thinking_count": len(e.thinking_actors),
                      "thinking_actors": e.thinking_actors, **self._limit_fields()},
            "objects": [w._obj_view(o).model_dump(mode="json") for o in w.objects.values()],
            "actors": actors,
            "relationships": {
                aid: {t: r.model_dump(mode="json", exclude={"recent_key_events"})
                      for t, r in e.agents[aid].relationships.rels.items()}
                for aid in e.order
            },
            "orders": [o.model_dump(mode="json") for o in w.orders if o.fulfilled_by is None],
            "events": list(self.timeline)[-120:],
            "metrics": self._metrics_view(),
            "sessions": [{"id": s.id, "participants": s.participants, "topic": s.topic}
                         for s in e.sessions.values()],
        }

    def agent_detail(self, aid: str) -> dict[str, Any] | None:
        e = self.engine
        agent = e.agents.get(aid)
        if agent is None:
            return None
        w = e.world
        a = w.actors[aid]
        spec = agent.spec
        mem = agent.memory.records
        shops = [
            {
                "id": o.id, "name": o.name, "kind": o.kind,
                "open": bool(o.data.get("open", True)),
                "stock": dict(sorted(o.data.get("stock", {}).items())),
                "prices": {item: w.shop_price(o, item)
                           for item in sorted(set(o.data.get("stock", {}))
                                              | set(o.data.get("prices", {})))},
                **({"treasury": o.data.get("treasury", 0)} if o.kind == "guild_counter" else {}),
            }
            for o in w.objects.values()
            if o.kind in ("shop_counter", "guild_counter") and o.owner == aid
        ]
        return {
            "id": aid, "name": spec.name, "age": spec.age, "occupation": spec.occupation,
            "profession": spec.profession, "skills": list(spec.skills),
            "traits": spec.traits, "persona": spec.persona,
            "activity": a.activity, "pos": list(a.pos), "facing": a.facing,
            "fainted": self._is_fainted(aid),
            "needs": {"hunger": round(a.hunger, 1), "stamina": round(a.stamina, 1),
                      "mood": round(a.mood, 1), "hp": round(a.hp, 1)},
            "equipment": {k: dict(v) for k, v in a.equipment.items()},
            "shops": shops,
            "inventory": dict(sorted(a.inventory.items())),
            "routine": agent.routine.model_dump(mode="json") if agent.routine else None,
            "plan": [b.model_dump(mode="json") for b in agent.plan],
            "recent_memories": [
                {"id": r.id, "tick": r.tick, "kind": r.kind, "text": r.text,
                 "importance": r.importance}
                for r in mem if r.kind not in ("reflection", "goal")
            ][-15:][::-1],
            "reflections": [{"id": r.id, "tick": r.tick, "text": r.text}
                            for r in agent.memory.of_kind("reflection")][-6:][::-1],
            "goals": list(agent.goals),
            "goal_details": [
                {"text": r.text.removeprefix("삶의 목표: "), "evidence_count": len(r.evidence_ids),
                 "tick": r.tick}
                for r in agent.memory.of_kind("goal")[-3:]
            ],
            "rumors": [
                {"id": r.id, "tick": r.tick, "text": rumor_text(r),
                 "source_id": r.source_actor_id,
                 "source": "신의 속삭임" if r.source_actor_id in (None, "rumor", "god")
                 else e.agents[r.source_actor_id].spec.name
                 if r.source_actor_id in e.agents else r.source_actor_id,
                 "confidence": round(r.confidence, 3),
                 "about": [s for s in r.subjects if s in e.agents and s != r.source_actor_id]}
                for r in agent.rumors()[-10:][::-1]
            ],
            "semantic_facts": agent.semantic_facts[-6:],
            "relationships": {t: r.model_dump(mode="json")
                              for t, r in agent.relationships.rels.items()},
            "incoming": {  # other -> this villager (for the graph's reverse edges)
                oid: e.agents[oid].relationships.rels[aid].model_dump(
                    mode="json", exclude={"recent_key_events"})
                for oid in e.order if oid != aid and aid in e.agents[oid].relationships.rels
            },
            "world_notes": list(agent.world_notes),
            "counters": dict(agent.counters),
            "in_conversation": aid in e.actor_session,
        }

    # ------------------------------------------------------------ saves
    ALL_JOB_KINDS = ("choose_routine", "plan_day", "reflect", "reflect_threshold", "weekly",
                     "turn")

    async def _quiesce_all(self) -> None:
        """(lock held) cancel every in-flight cognition and wait for it to settle."""
        e = self.engine
        for aid in list(e.jobs):
            e._cancel_job(aid, self.ALL_JOB_KINDS)
        await e.quiesce()

    def save_info(self) -> dict[str, Any]:
        e = self.engine
        w = e.world
        store = e.store
        return {
            "save_path": store.path if store is not None else None,
            "persistent": store is not None and store.path != ":memory:",
            "current_tick": w.tick, "current_day": w.day, "current_minute": w.minute,
            "checkpoints": store.list_checkpoints(50, e.config.ticks_per_day)
            if store is not None else [],
        }

    async def save_checkpoint(self, label: str = "") -> dict[str, Any]:
        e = self.engine
        if e.store is None:
            raise ValueError("memory-only run: start the server with a save path to checkpoint")
        async with self._lock:
            await e.quiesce()  # finish (not cancel) in-flight thoughts: nothing is lost
            cid = e.checkpoint(label.strip() or MANUAL_LABEL)
            self.replay.flush()  # buffered frames up to the save point are on disk too
            self._reset_hold()
            self._flush(force=True)
        meta = e.store.checkpoint_meta(cid, e.config.ticks_per_day)
        assert meta is not None
        return meta

    async def restore_checkpoint(self, *, tick: int | None = None,
                                 checkpoint_id: int | None = None) -> dict[str, Any]:
        e = self.engine
        if e.store is None:
            raise ValueError("memory-only run has no checkpoints")
        async with self._lock:
            if checkpoint_id is not None:
                at, data = e.store.load_checkpoint_id(checkpoint_id)
            elif tick is not None:
                at, data = e.store.load_checkpoint(at_or_before=tick)
            else:
                raise ValueError("tick or id required")
            await self._quiesce_all()
            self._drop_unsaved()
            # Linear history: everything after the restored point (log, memory archive,
            # tape *and* later checkpoints) belongs to an abandoned future and is removed,
            # so no checkpoint can ever exist without the archive rows it depends on.
            e.store.truncate_after(at)
            self.replay.truncate_after(at)
            e.load_snapshot(data)
            self._after_rewind()
        return {"tick": e.world.tick, "day": e.world.day, "minute": e.world.minute}

    async def reset_village(self) -> dict[str, Any]:
        e = self.engine
        async with self._lock:
            await self._quiesce_all()
            fresh = e.fresh_snapshot()
            self._drop_unsaved()
            if e.store is not None:
                e.store.reset_all()  # a new village: no checkpoints, log or archive survive
            self.replay.reset()
            e.load_snapshot(fresh)
            self._after_rewind()
        return {"tick": e.world.tick, "day": e.world.day, "minute": e.world.minute}

    def _drop_unsaved(self) -> None:
        """(lock held) discard what the abandoned timeline had not written yet - events,
        archive rows and LLM tape after the restore point must never reach the store."""
        e = self.engine
        e.pending_events = []
        e.pending_archive = []
        e.router.drain_tape()

    def _after_rewind(self) -> None:
        """(lock held) drop stale buffers and push a full snapshot to every client."""
        self._buffer = []
        self._seen = 0
        self._last_activity = {}
        self._step_budget = 0
        self.timeline.clear()
        # speech/events of the abandoned timeline must never leak into new replay frames
        self._speech.clear()
        self._fainted_until.clear()
        self._tick_events = []
        self.engine.reset_barrier()
        self._held = False
        self._last_thinking = []
        self._new_window()  # the clock jumped: count days from the restored point
        snap = self.snapshot()  # carries thinking_count / thinking_actors
        for q in list(self.clients):
            q.put_nowait(snap)

    # ------------------------------------------------------------ provider hot-swap
    async def switch_llm(self, llm: Any, embedder: Any, *, pricing: Any = None,
                         timeout: float | None = None, retries: int | None = None,
                         max_concurrency: int | None = None) -> None:
        """Swap the live LLM + embedder under the engine lock.

        In-flight cognition is cancelled (and re-queued), awaited, and only then are the old
        HTTP clients closed, so no request ever runs against a closed client.
        """
        from ..llm.protocol import Observability, RouterStats, SharedRateGate

        async with self._lock:
            e = self.engine
            for aid in list(e.jobs):
                e._cancel_job(aid, ("choose_routine", "plan_day", "reflect",
                                    "reflect_threshold", "weekly", "turn"))
            await e.quiesce()
            router = e.router
            old_llm, old_embedder = router.llm, e.embedder
            router.llm = llm
            if timeout is not None:
                router.timeout = timeout
            if retries is not None:
                router.retries = retries
            e.update_barrier_timeout(router.timeout, router.retries)
            if max_concurrency is not None:  # safe: nothing is in flight after quiesce()
                router.gate = SharedRateGate(max_concurrency)
            router.obs = Observability(pricing)  # fresh telemetry for the new profile
            router.stats = RouterStats()
            if embedder is not old_embedder:
                e.embedder = embedder
                for agent in e.agents.values():
                    agent.memory.set_embedder(embedder)
            self._emit({"type": "llm_switched",
                        "provider": getattr(llm, "provider", "demo"),
                        "model": getattr(llm, "model", "?")})
            self._reset_hold()
            self._flush(force=True)
        if old_llm is not llm:
            aclose = getattr(old_llm, "aclose", None)
            if aclose is not None:
                await aclose()
        if old_embedder is not embedder:
            drain = getattr(old_embedder, "drain", None)
            if drain is not None:
                await drain()  # threads of cancelled aembed calls finish before close
            close = getattr(old_embedder, "close", None)
            if close is not None:
                close()

    # ------------------------------------------------------------ commands
    async def command(self, msg: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:  # never interleave with a running engine.step()
            return await self._command_locked(msg)

    async def _command_locked(self, msg: dict[str, Any]) -> dict[str, Any]:
        cmd = msg.get("cmd")
        e = self.engine
        try:
            if cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                if self.limit_reached:
                    self._new_window()  # ▶ after the limit grants another N days
                self.paused = False
                self._wake.set()
            elif cmd == "set_max_days":
                self.max_run_days = normalize_max_days(msg["days"])
                self.limit_reached = False  # re-evaluated against the new limit
                self._check_limit()
            elif cmd == "set_speed":
                speed = float(msg["speed"])
                if speed not in ALLOWED_SPEEDS:
                    raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}")
                self.speed = speed
                self.pacer.set_speed(speed)
            elif cmd == "step":
                if self.limit_reached:
                    raise ValueError("최대 실행 일수에 도달했습니다. ▶ 재생을 눌러 계속하세요")
                self._step_budget += max(1, min(1440, int(msg.get("ticks", 1))))
                self._wake.set()
            elif cmd == "checkpoint":
                if e.store is None:
                    raise ValueError("no save store configured")
                await e.quiesce()
                e.checkpoint(str(msg.get("label") or "").strip()[:80] or MANUAL_LABEL)
                self._reset_hold()
            elif cmd == "god_weather":
                e.set_weather(Weather(str(msg["weather"]).lower()))
            elif cmd == "god_spawn":
                self._spawn(str(msg["kind"]), int(msg["x"]), int(msg["y"]))
            elif cmd == "god_event":
                text = str(msg["text"]).strip()
                if not text:
                    raise ValueError("empty event text")
                e.inject_event(text[:200])
                if e._blocking_jobs():
                    self._emit(self._clock_msg())  # mid-barrier: the wait keeps its deadline
                else:
                    self._reset_hold()
            elif cmd == "god_rumor":
                target, about = str(msg["target"]), str(msg["about"])
                if target not in e.agents or about not in e.agents or target == about:
                    raise ValueError("unknown or identical villagers")
                e.inject_rumor(target, about, str(msg["text"])[:200])
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except (KeyError, ValueError, TypeError) as exc:
            return {"type": "ack", "cmd": cmd, "ok": False, "error": str(exc)}
        self._capture()
        for d in e.drain_deltas():
            payload = d.model_dump(mode="json")
            if payload["type"] == "object_change":
                obj = e.world.objects.get(payload["object_id"])
                if obj is not None:
                    payload["data"] = e.world._obj_view(obj).data
            self._emit(payload)
        if cmd in ("pause", "resume", "set_speed", "set_max_days", "god_weather"):
            self._emit(self._clock_msg())
        self._flush(force=True)
        return {"type": "ack", "cmd": cmd, "ok": True}

    def _spawn(self, kind: str, x: int, y: int) -> None:
        if kind not in SPAWN_KINDS:
            raise ValueError(f"spawn kind must be one of {sorted(SPAWN_KINDS)}")
        obj_kind, state = SPAWN_KINDS[kind]
        e = self.engine
        before = set(e.world.objects)
        e.spawn_resource(obj_kind, (x, y), state)
        if kind == "seeds":
            (oid,) = set(e.world.objects) - before
            e.world.objects[oid].data["items"] = {"turnip_seed": 4, "wheat_seed": 2}

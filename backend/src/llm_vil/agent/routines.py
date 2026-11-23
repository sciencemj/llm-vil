"""Deterministic routine executors.

The LLM picks one *routine* with enum-constrained parameters; the executor expands it,
one call at a time, into atomic affordances (``move_to`` + object actions). Executors are
pure functions of ``(choice, progress, context)``; ``progress`` is a JSON-able dict kept in
agent state so a routine survives snapshot/restore mid-way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field

from ..llm.protocol import stable_unit
from ..schemas.agent import AgentSpec, RoutineChoice
from ..schemas.base import Mutable
from ..schemas.config import VillageConfig
from ..schemas.world import AtomicAction, TilePos, WorldView

MAX_ROUTINE_STEPS = 60
SEED_PREFERENCE = ("turnip_seed", "wheat_seed")


class KnownObject(Mutable):
    """Per-agent spatial memory entry (last-seen state of a discovered object)."""

    id: str
    kind: str
    pos: tuple[int, int]
    state: str
    owner: str | None = None
    name: str = ""
    items: dict[str, int] = Field(default_factory=dict)  # chests: last-seen contents
    last_seen: int = 0
    version: int = 0  # world object version when last seen


@dataclass(frozen=True)
class RoutineContext:
    spec: AgentSpec
    config: VillageConfig
    view: WorldView
    known: dict[str, KnownObject]


def _pos(ctx: RoutineContext) -> tuple[int, int]:
    p = ctx.view.self_actor.pos
    return (p.x, p.y)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _state(ctx: RoutineContext, oid: str) -> str | None:
    for o in ctx.view.objects:
        if o.id == oid:
            return o.state
    k = ctx.known.get(oid)
    return k.state if k else None


def _approach(ctx: RoutineContext, oid: str, action: AtomicAction) -> AtomicAction:
    k = ctx.known.get(oid)
    if k is not None and _dist(_pos(ctx), k.pos) <= 1:
        return action
    return AtomicAction(kind="move_to", target_id=oid)


def _once(progress: dict[str, Any], ctx: RoutineContext, oid: str | None,
          action: AtomicAction) -> AtomicAction | None:
    if progress.get("issued"):
        return None
    step = _approach(ctx, oid, action) if oid else action
    if step.kind != "move_to":
        progress["issued"] = True
    return step


def _nearest(ctx: RoutineContext, kind: str, states: tuple[str, ...],
             area: str | None = None, skip: list[str] | None = None) -> str | None:
    here = _pos(ctx)
    area_def = ctx.config.area(area) if area and area not in ("none", "any") else None
    cands = []
    for k in ctx.known.values():
        if k.kind != kind or (skip and k.id in skip):
            continue
        if _state(ctx, k.id) not in states:
            continue
        if area_def is not None and not area_def.contains(*k.pos):
            continue
        cands.append((_dist(here, k.pos), k.id))
    return min(cands)[1] if cands else None


# ---------------------------------------------------------------- executors
def _tend_farm(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    plots = list(ctx.spec.plots) if c.target in ("my_plots", "none") else [c.target]
    inv = ctx.view.self_actor.inventory
    raining = ctx.view.weather == "rain"
    for pid in plots:
        st = _state(ctx, pid)
        act: AtomicAction | None = None
        if st == "ready":
            act = AtomicAction(kind="harvest", target_id=pid)
        elif st == "untilled":
            act = AtomicAction(kind="till", target_id=pid)
        elif st == "tilled":
            seed = next((s for s in SEED_PREFERENCE if inv.get(s, 0) > 0), None)
            if seed == "turnip_seed" and inv.get("wheat_seed", 0) > 0 and stable_unit(
                ctx.spec.id, pid, ctx.view.day
            ) < 0.4:
                seed = "wheat_seed"
            if seed:
                act = AtomicAction(kind="plant", target_id=pid, item=seed)
        elif st == "planted" and not raining:
            act = AtomicAction(kind="water", target_id=pid)
        if act is not None:
            return _approach(ctx, pid, act)
    return None


def _chop_wood(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    wood = ctx.view.self_actor.inventory.get("wood", 0)
    start = p.setdefault("start_wood", wood)
    if wood - start >= max(1, c.qty):
        return None
    tid = _nearest(ctx, "tree", ("mature",), area=c.target)
    if tid is None:
        return None
    return _approach(ctx, tid, AtomicAction(kind="chop", target_id=tid))


def _forage(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    berries = ctx.view.self_actor.inventory.get("berry", 0)
    start = p.setdefault("start_berry", berries)
    if berries - start >= 5:
        return None
    if c.target in ctx.known and _state(ctx, c.target) == "ripe":
        bid: str | None = c.target
    else:
        bid = _nearest(ctx, "berry_bush", ("ripe",))
    if bid is None:
        return None
    return _approach(ctx, bid, AtomicAction(kind="forage", target_id=bid))


def _cook(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(kind="cook", target_id=c.target, item=c.item))


def _eat(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, None, AtomicAction(kind="eat", item=c.item))


def _store_or_take(c: RoutineChoice, p: dict[str, Any],
                   ctx: RoutineContext) -> AtomicAction | None:
    if c.op not in ("store", "take"):
        return None  # invalid op is a no-op, never silently remapped
    kind = c.op
    return _once(p, ctx, c.target,
                 AtomicAction(kind=kind, target_id=c.target, item=c.item, qty=max(1, c.qty)))


def _fulfill(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target,
                 AtomicAction(kind="fulfill_order", target_id=c.target, item=c.item))


def _trade(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(kind="trade", target_id=c.target, item=c.item,
                                                op=c.op, qty=max(1, c.qty)))


def _sleep(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(kind="sleep", target_id=c.target))


def _area_tile(ctx: RoutineContext, area_id: str, salt: Any) -> TilePos:
    area = ctx.config.area(area_id) or ctx.config.area(ctx.spec.home_area)
    assert area is not None
    x, y, w, h = area.rect
    u = stable_unit(ctx.spec.id, ctx.view.tick, salt)
    v = stable_unit(salt, ctx.view.tick, ctx.spec.id)
    return TilePos(x=x + int(u * w), y=y + int(v * h))


def _rest(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    phase = p.get("phase", 0)
    p["phase"] = phase + 1
    if phase == 0:
        area = ctx.config.area(c.target) or ctx.config.area(ctx.spec.home_area)
        assert area is not None
        if area.contains(*_pos(ctx)):
            p["phase"] = 2
            return AtomicAction(kind="rest", ticks=30)
        return AtomicAction(kind="move_to", target_pos=_area_tile(ctx, area.id, "rest"))
    if phase == 1:
        return AtomicAction(kind="rest", ticks=30)
    return None


def _wander(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    phase = p.get("phase", 0)
    if phase >= 4:
        return None
    p["phase"] = phase + 1
    if phase % 2 == 0:
        target = c.target if ctx.config.area(c.target) else ctx.spec.home_area
        return AtomicAction(kind="move_to", target_pos=_area_tile(ctx, target, phase))
    return AtomicAction(kind="wait", ticks=10)


def _visit(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    if p.get("issued"):
        return None
    here = _pos(ctx)
    for a in ctx.view.actors:
        if a.id == c.target and _dist(here, (a.pos.x, a.pos.y)) <= 2:
            p["issued"] = True
            return AtomicAction(kind="talk", target_id=c.target, item=c.topic or "안부")
    hops = p.get("hops", 0)
    if hops >= 6:
        return None
    p["hops"] = hops + 1
    return AtomicAction(kind="move_to", target_id=c.target)


def resolve_targets(choice: RoutineChoice, ctx: RoutineContext) -> list[str]:
    """Known world objects a routine choice will act on (for stale-view validation)."""
    if choice.routine == "tend_farm":
        plots = ctx.spec.plots if choice.target in ("my_plots", "none") else [choice.target]
        return [p for p in plots if p in ctx.known]
    if choice.routine == "chop_wood":
        tid = _nearest(ctx, "tree", ("mature",), area=choice.target)
        return [tid] if tid else []
    if choice.routine == "forage" and choice.target not in ctx.known:
        bid = _nearest(ctx, "berry_bush", ("ripe",))
        return [bid] if bid else []
    return [choice.target] if choice.target in ctx.known else []


EXECUTORS = {
    "tend_farm": _tend_farm,
    "chop_wood": _chop_wood,
    "forage": _forage,
    "cook_meal": _cook,
    "eat_food": _eat,
    "store_or_take": _store_or_take,
    "visit_and_talk": _visit,
    "post_or_fulfill_order": _fulfill,
    "trade": _trade,
    "sleep": _sleep,
    "rest": _rest,
    "wander": _wander,
}


def next_step(choice: RoutineChoice, progress: dict[str, Any],
              ctx: RoutineContext) -> AtomicAction | None:
    """Next atomic action for ``choice``, or ``None`` when the routine is complete."""
    steps = progress.get("steps", 0)
    if steps >= MAX_ROUTINE_STEPS:
        return None
    progress["steps"] = steps + 1
    return EXECUTORS[choice.routine](choice, progress, ctx)

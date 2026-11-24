"""Deterministic routine executors.

The LLM picks one *routine* with enum-constrained parameters; the executor expands it,
one call at a time, into atomic affordances (``move_to`` + object actions). Executors are
pure functions of ``(choice, progress, context)``; ``progress`` is a JSON-able dict kept in
agent state so a routine survives snapshot/restore mid-way.
"""

from __future__ import annotations

import math
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
    items: dict[str, int] = Field(default_factory=dict)  # chests / shop stock: last seen
    prices: dict[str, int] = Field(default_factory=dict)  # shop counters: last-seen prices
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


# ---------------------------------------------------------------- Phase 2 executors
MOVING_KINDS = frozenset({"monster"})
COUNTER_KINDS = frozenset({"shop_counter", "guild_counter"})
HP_RETREAT = 45.0


def known_price(config: VillageConfig, known: KnownObject | None, item: str) -> int:
    """Last-seen shop price, else the item's base value (what the world falls back to)."""
    if known is not None and known.prices.get(item):
        return int(known.prices[item])
    it = config.items.get(item)
    return max(1, it.value) if it else 1


def price_band(config: VillageConfig, item: str) -> tuple[int, int]:
    it = config.items.get(item)
    v = max(1, it.value) if it else 1
    s = config.shops
    return max(1, math.floor(v * s.price_min)), max(1, math.ceil(v * s.price_max))


def bounty_reward(config: VillageConfig, species: str) -> int:
    mon = config.monsters.get(species)
    return 8 + round((mon.power if mon else 0.0) * 40)


def request_reward(config: VillageConfig, item: str, qty: int) -> int:
    it = config.items.get(item)
    v = max(1, it.value) if it else 1
    return max(2, round(v * max(1, qty) * 1.25))


def recipe_needs(inputs: dict[str, int], fuel: dict[str, int], fuel_any: list[str],
                 inv: dict[str, int]) -> dict[str, int] | None:
    """Agent-side mirror of the world's recipe check (inputs + fuel + one ``fuel_any``)."""
    need = dict(inputs)
    for item, n in fuel.items():
        need[item] = need.get(item, 0) + n
    if fuel_any:
        f = next((f for f in fuel_any if inv.get(f, 0) >= need.get(f, 0) + 1), None)
        if f is None:
            return None
        need[f] = need.get(f, 0) + 1
    return None if any(inv.get(i, 0) < n for i, n in need.items()) else need


def _repeat(p: dict[str, Any], ctx: RoutineContext, oid: str | None, action: AtomicAction,
            times: int) -> AtomicAction | None:
    """Issue ``action`` up to ``times`` (approaching first); a rejection ends the routine."""
    if p.get("done", 0) >= max(1, times):
        return None
    step = _approach(ctx, oid, action) if oid else action
    if step.kind != "move_to":
        p["done"] = p.get("done", 0) + 1
    return step


def _craft(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    recipe = ctx.config.recipes.get(c.item)
    if recipe is not None and p.get("done", 0) and recipe_needs(
        recipe.inputs, recipe.fuel, recipe.fuel_any, ctx.view.self_actor.inventory
    ) is None:
        return None  # ran out of ingredients: the batch is done
    return _repeat(p, ctx, c.target, AtomicAction(kind="craft", target_id=c.target,
                                                  item=c.item), min(3, max(1, c.qty)))


def _stock_shop(c: RoutineChoice, p: dict[str, Any],
                ctx: RoutineContext) -> AtomicAction | None:
    phase = p.get("phase", 0)
    inv = ctx.view.self_actor.inventory
    k = ctx.known.get(c.target)
    if phase == 0:
        qty = min(max(1, c.qty), inv.get(c.item, 0))
        if qty < 1:
            return None
        step = _approach(ctx, c.target, AtomicAction(kind="stock_shop", target_id=c.target,
                                                     item=c.item, qty=qty))
        if step.kind != "move_to":
            p["phase"] = 1
            p["before"] = (k.items.get(c.item, 0) if k else 0)
        return step
    if phase == 1:
        p["phase"] = 2
        # simple market-making: unsold piles get cheaper, sold-out goods dearer
        before = p.get("before", 0)
        lo, hi = price_band(ctx.config, c.item)
        cur = known_price(ctx.config, k, c.item)
        new = cur - 1 if before >= 6 else cur + 1 if before == 0 else cur
        new = max(lo, min(hi, new))
        if new != cur:
            return AtomicAction(kind="set_prices", target_id=c.target, item=c.item, amount=new)
    return None


def _buy_from_shop(c: RoutineChoice, p: dict[str, Any],
                   ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(kind="buy_from_shop", target_id=c.target,
                                                item=c.item, qty=max(1, c.qty)))


def _gather(kind: str, action: str, ready: str, area: str, item: str):  # type: ignore[no-untyped-def]
    def run(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
        have = ctx.view.self_actor.inventory.get(item, 0)
        start = p.setdefault("start", have)
        if have - start >= max(2, c.qty * 2) or p.get("done", 0) >= max(1, c.qty):
            return None
        if c.target in ctx.known and _state(ctx, c.target) == ready:
            oid: str | None = c.target
        else:
            zone = c.target if ctx.config.area(c.target) else area
            oid = _nearest(ctx, kind, (ready,), area=zone)
            if oid is None:
                oid = _nearest(ctx, kind, (ready,))
        if oid is None:
            return None
        step = _approach(ctx, oid, AtomicAction(kind=action, target_id=oid))  # type: ignore[arg-type]
        if step.kind != "move_to":
            p["done"] = p.get("done", 0) + 1
        return step
    return run


_mine = _gather("ore_node", "mine", "rich", "quarry", "iron_ore")
_gather_herb = _gather("herb_patch", "gather_herb", "ripe", "pond", "herb")


def _hunt(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    """Walk into Mosswood, engage the nearest roaming monster (bounty species first)."""
    me = ctx.view.self_actor
    if me.needs.hp <= HP_RETREAT or p.get("fights", 0) >= max(1, c.qty):
        return None
    here = _pos(ctx)
    species = c.item if c.item in ctx.config.monsters else None
    seen = [o for o in ctx.view.objects if o.kind == "monster" and o.state == "roaming"]
    if species and any(o.data.get("species") == species for o in seen):
        seen = [o for o in seen if o.data.get("species") == species]
    if c.target in {o.id for o in seen}:
        seen = [o for o in seen if o.id == c.target]
    if seen:
        mon = min(seen, key=lambda o: (_dist(here, (o.pos.x, o.pos.y)), o.id))
        if _dist(here, (mon.pos.x, mon.pos.y)) <= 1:
            p["fights"] = p.get("fights", 0) + 1
            p["hops"] = 0
            return AtomicAction(kind="fight", target_id=mon.id)
        hops = p.get("chase", 0)
        if hops >= 8:
            return None
        p["chase"] = hops + 1
        return AtomicAction(kind="move_to", target_id=mon.id)
    hops = p.get("hops", 0)
    if hops >= 5:
        return None
    p["hops"] = hops + 1
    return AtomicAction(kind="move_to", target_pos=_area_tile(ctx, "mosswood", ("hunt", hops)))


def _sell_drops(c: RoutineChoice, p: dict[str, Any],
                ctx: RoutineContext) -> AtomicAction | None:
    inv = ctx.view.self_actor.inventory
    sold = p.setdefault("sold", [])
    items = [c.item] if c.item not in ("none", "all") else []
    items += [i for i, n in sorted(inv.items())
              if n > 0 and (it := ctx.config.items.get(i)) is not None and it.drop
              and i not in items]
    guild = ctx.known.get(c.target)
    cap = ctx.config.shops.guild_stock_cap
    for item in items:
        room = cap - (guild.items.get(item, 0) if guild else 0)
        if item in sold or inv.get(item, 0) < 1 or room < 1:
            continue
        step = _approach(ctx, c.target, AtomicAction(kind="sell_drops", target_id=c.target,
                                                     item=item, qty=min(inv[item], room)))
        if step.kind != "move_to":
            sold.append(item)
        return step
    return None


def _post_bounty(c: RoutineChoice, p: dict[str, Any],
                 ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(
        kind="post_bounty", target_id=c.target, item=c.item, qty=1,
        amount=bounty_reward(ctx.config, c.item)))


def _post_request(c: RoutineChoice, p: dict[str, Any],
                  ctx: RoutineContext) -> AtomicAction | None:
    qty = max(1, min(5, c.qty))
    return _once(p, ctx, c.target, AtomicAction(
        kind="post_request", target_id=c.target, item=c.item, qty=qty,
        amount=request_reward(ctx.config, c.item, qty)))


def _repair_tool(c: RoutineChoice, p: dict[str, Any],
                 ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, c.target, AtomicAction(kind="repair_tool", target_id=c.target,
                                                item=c.item))


def _use_potion(c: RoutineChoice, p: dict[str, Any],
                ctx: RoutineContext) -> AtomicAction | None:
    return _once(p, ctx, None, AtomicAction(kind="eat", item=c.item))


def _run_stall(c: RoutineChoice, p: dict[str, Any], ctx: RoutineContext) -> AtomicAction | None:
    """Merchant: buy a lot from the travelling caravan, then shelve it in the own shop."""
    phase = p.get("phase", 0)
    if phase == 0:
        step = _approach(ctx, c.target, AtomicAction(kind="trade", target_id=c.target,
                                                     item=c.item, op="buy",
                                                     qty=max(1, c.qty)))
        if step.kind != "move_to":
            p["phase"] = 1
        return step
    if phase == 1 and ctx.spec.shop_id:
        inv = ctx.view.self_actor.inventory
        if inv.get(c.item, 0) < 1:
            return None
        step = _approach(ctx, ctx.spec.shop_id, AtomicAction(
            kind="stock_shop", target_id=ctx.spec.shop_id, item=c.item, qty=inv[c.item]))
        if step.kind != "move_to":
            p["phase"] = 2
        return step
    return None


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
    if choice.routine in ("mine", "gather_herb") and choice.target not in ctx.known:
        kind, ready = ("ore_node", "rich") if choice.routine == "mine" else ("herb_patch", "ripe")
        oid = _nearest(ctx, kind, (ready,))
        return [oid] if oid else []
    k = ctx.known.get(choice.target)
    # counters change on every sale and monsters wander: the world re-validates those
    if k is None or k.kind in COUNTER_KINDS or k.kind in MOVING_KINDS:
        return []
    return [choice.target]


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
    "craft": _craft,
    "stock_shop": _stock_shop,
    "buy_from_shop": _buy_from_shop,
    "mine": _mine,
    "gather_herb": _gather_herb,
    "hunt": _hunt,
    "sell_drops": _sell_drops,
    "post_bounty": _post_bounty,
    "post_request": _post_request,
    "repair_tool": _repair_tool,
    "use_potion": _use_potion,
    "run_stall": _run_stall,
}


def next_step(choice: RoutineChoice, progress: dict[str, Any],
              ctx: RoutineContext) -> AtomicAction | None:
    """Next atomic action for ``choice``, or ``None`` when the routine is complete."""
    steps = progress.get("steps", 0)
    if steps >= MAX_ROUTINE_STEPS:
        return None
    progress["steps"] = steps + 1
    return EXECUTORS[choice.routine](choice, progress, ctx)

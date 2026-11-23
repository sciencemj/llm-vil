"""World state and the single validity gate ``World.apply``."""

from __future__ import annotations

import random
from typing import Any

from pydantic import Field

from ..schemas.base import Mutable
from ..schemas.config import VillageConfig
from ..schemas.world import (
    ActionResult,
    ActorSnapshot,
    AtomicAction,
    NeedsState,
    OrderState,
    TilePos,
    Weather,
    WorldEvent,
    WorldObjectState,
    WorldView,
)
from .affordances import AffordanceRegistry
from .grid import Tile, TileGrid, facing_toward

OBJECT_AFFORDANCES = frozenset(
    {"chop", "till", "plant", "water", "harvest", "forage", "cook", "sleep", "store", "take",
     "fulfill_order", "trade"}
)
ACTION_DURATIONS = {"eat": 5, "give_item": 0}
DEFAULT_REST_TICKS = 30
DEFAULT_WAIT_TICKS = 10


class ObjState(Mutable):
    id: str
    kind: str
    pos: tuple[int, int]
    state: str
    owner: str | None = None
    name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    version: int = 0


class ActiveAction(Mutable):
    action: AtomicAction
    start_tick: int
    end_tick: int
    path: list[tuple[int, int]] = Field(default_factory=list)
    next_step_tick: int = 0
    step_ticks: int = 1


class ActorState(Mutable):
    id: str
    name: str
    pos: tuple[int, int]
    facing: int = 90
    activity: str = "sleeping"
    hunger: float = 20.0
    stamina: float = 90.0
    mood: float = 65.0
    inventory: dict[str, int] = Field(default_factory=dict)
    action: ActiveAction | None = None
    version: int = 0


class Order(Mutable):
    id: str
    item: str
    qty: int
    reward: int
    posted_day: int
    expires_day: int
    fulfilled_by: str | None = None


def _cheb(a: Tile, b: Tile) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


PRIVATE_DATA_KEYS = frozenset({"missing"})  # never exposed through WorldView


def _copy_data(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = [dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return lo if v < lo else hi if v > hi else v


class World:
    """Authoritative village state. Every mutation goes through ``apply`` / ``tick_once`` /
    explicit engine or God-mode hooks, and bumps the monotonic ``world_version``."""

    def __init__(self, config: VillageConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.registry = AffordanceRegistry(config)
        self.rng = random.Random(config.seed if seed is None else seed)
        self.tick: int = config.start_minute
        self.world_version = 0
        self.weather = Weather.sunny
        self.merchant_present = False
        self.order_seq = 0
        self.spawn_seq = 0
        self.orders: list[Order] = []
        self.objects: dict[str, ObjState] = {}
        for o in config.objects:
            kind = config.kinds[o.kind]
            data = _copy_data(o.data)
            if o.kind == "chest":
                data.setdefault("items", {})
                data.setdefault("missing", [])
            if kind.growth is not None:
                data.setdefault("stage_days", 0)
            self.objects[o.id] = ObjState(
                id=o.id, kind=o.kind, pos=o.pos, state=o.state or kind.initial_state,
                owner=o.owner, name=o.name or o.id, data=data,
            )
        self.actors: dict[str, ActorState] = {
            v.id: ActorState(id=v.id, name=v.name, pos=v.start_pos, inventory=dict(v.inventory))
            for v in config.villagers
        }
        self._food = frozenset(k for k, it in config.items.items() if it.food > 0)
        self._rebuild_static()
        self._post_orders()

    # ------------------------------------------------------------------ time / helpers
    @property
    def day(self) -> int:
        return self.tick // self.config.ticks_per_day

    @property
    def minute(self) -> int:
        return self.tick % self.config.ticks_per_day

    def _rebuild_static(self) -> None:
        blocked: set[Tile] = set()
        for x, y, w, h in self.config.water:
            blocked.update((xx, yy) for xx in range(x, x + w) for yy in range(y, y + h))
        self.opaque: set[Tile] = set()
        for o in self.objects.values():
            kind = self.config.kinds[o.kind]
            if kind.blocking:
                blocked.add(o.pos)
            if kind.opaque:
                self.opaque.add(o.pos)
        self.grid = TileGrid(self.config.width, self.config.height, blocked)

    def _bump(self, thing: ObjState | ActorState) -> None:
        self.world_version += 1
        thing.version = self.world_version

    def _event(self, kind: str, *, actor: str | None = None, target: str | None = None,
               pos: Tile | None = None, importance: int = 2, witnesses: list[str] | None = None,
               **data: Any) -> WorldEvent:
        return WorldEvent(
            tick=self.tick, kind=kind, actor_id=actor, target_id=target,
            pos=TilePos(x=pos[0], y=pos[1]) if pos else None, data=data,
            importance=importance, witnesses=witnesses or [],
        )

    def can_perceive(self, actor_id: str, pos: Tile) -> bool:
        a = self.actors[actor_id]
        if a.activity == "sleeping":
            return False
        return _cheb(a.pos, pos) <= self.config.perception_radius and TileGrid.line_of_sight(
            a.pos, pos, self.opaque
        )

    def _witnesses(self, pos: Tile, exclude: set[str]) -> list[str]:
        return [aid for aid in self.actors if aid not in exclude and self.can_perceive(aid, pos)]

    def mature_tree_count(self) -> int:
        return sum(1 for o in self.objects.values() if o.kind == "tree" and o.state == "mature")

    def object_version(self, object_id: str) -> int | None:
        o = self.objects.get(object_id)
        return o.version if o else None

    # ------------------------------------------------------------------ perception
    def _obj_view(self, o: ObjState) -> WorldObjectState:
        return WorldObjectState.model_construct(
            id=o.id, kind=o.kind, pos=TilePos.model_construct(x=o.pos[0], y=o.pos[1]),
            state=o.state, owner=o.owner, name=o.name, version=o.version,
            data=_copy_data({k: v for k, v in o.data.items() if k not in PRIVATE_DATA_KEYS}),
        )

    def actor_snapshot(self, actor_id: str) -> ActorSnapshot:
        a = self.actors[actor_id]
        return ActorSnapshot.model_construct(
            id=a.id, name=a.name, pos=TilePos.model_construct(x=a.pos[0], y=a.pos[1]),
            facing=a.facing, activity=a.activity,
            needs=NeedsState.model_construct(hunger=a.hunger, stamina=a.stamina, mood=a.mood),
            inventory=dict(a.inventory), version=a.version,
        )

    def perceive(self, actor_id: str) -> WorldView:
        a = self.actors[actor_id]
        r = self.config.perception_radius
        objs = [self._obj_view(o) for o in self.objects.values() if _cheb(o.pos, a.pos) <= r]
        others = [
            self.actor_snapshot(b.id)
            for b in self.actors.values()
            if b.id != actor_id and _cheb(a.pos, b.pos) <= r
            and TileGrid.line_of_sight(a.pos, b.pos, self.opaque)
        ]
        orders = [OrderState(**o.model_dump()) for o in self.orders if o.fulfilled_by is None]
        return WorldView.model_construct(
            world_version=self.world_version, tick=self.tick, day=self.day, minute=self.minute,
            weather=self.weather, self_actor=self.actor_snapshot(actor_id), objects=objs,
            actors=others, orders=orders, merchant_present=self.merchant_present,
        )

    # ------------------------------------------------------------------ validity gate
    def _reject(self, actor_id: str, action: AtomicAction, reason: str,
                events: list[WorldEvent] | None = None) -> ActionResult:
        return ActionResult(ok=False, actor_id=actor_id, action=action, reason=reason,
                            end_tick=self.tick, events=events or [])

    def apply(self, actor_id: str, action: AtomicAction) -> ActionResult:
        """Single validity gate for every atomic affordance (agents, engine, future humans)."""
        a = self.actors.get(actor_id)
        if a is None:
            return self._reject(actor_id, action, "unknown_actor")
        if action.kind == "talk":
            return self._reject(actor_id, action, "engine_only")
        if a.action is not None:
            return self._reject(actor_id, action, "busy")
        if a.activity == "sleeping" and action.kind != "wake":
            return self._reject(actor_id, action, "asleep")
        if action.kind == "wake" and a.activity != "sleeping":
            return self._reject(actor_id, action, "not_asleep")
        if a.activity == "in_conversation" and action.kind != "give_item":
            return self._reject(actor_id, action, "in_conversation")
        reason = self._check(a, action)
        if reason:
            # An owner reaching into a chest a thief emptied still notices the loss.
            obj = self.objects.get(action.target_id or "")
            if reason == "chest_empty" and obj is not None and obj.owner == a.id and (
                obj.data.get("missing") and _cheb(a.pos, obj.pos) <= 1
            ):
                return self._reject(actor_id, action, reason, self._discover_missing(a, obj))
            return self._reject(actor_id, action, reason)
        return self._start(a, action)

    def _crop_for_seed(self, seed: str | None) -> str | None:
        return next((name for name, c in self.config.crops.items() if c.seed == seed), None)

    def _check(self, a: ActorState, act: AtomicAction) -> str | None:
        k = act.kind
        inv = a.inventory
        if k in OBJECT_AFFORDANCES:
            obj = self.objects.get(act.target_id or "")
            if obj is None:
                return "no_target"
            aff = self.registry.get(obj.kind, k)
            if aff is None:
                return "not_afforded"
            if aff.from_states and obj.state not in aff.from_states:
                return f"state_{obj.state}"
            if _cheb(a.pos, obj.pos) > aff.reach:
                return "too_far"
            if aff.stamina_cost and a.stamina < aff.stamina_cost:
                return "too_tired"
            if k == "chop" and self.mature_tree_count() <= self.config.economy.forest_reserve:
                return "forest_reserve"
            if k == "plant":
                if self._crop_for_seed(act.item) is None:
                    return "bad_seed"
                if inv.get(act.item or "", 0) < 1:
                    return "no_seed"
            if k == "cook":
                recipe = self.config.recipes.get(act.item or "")
                if recipe is None:
                    return "bad_recipe"
                need = dict(recipe.inputs)
                for item, n in recipe.fuel.items():
                    need[item] = need.get(item, 0) + n
                if any(inv.get(i, 0) < n for i, n in need.items()):
                    return "missing_ingredients"
            if k == "store" and (act.qty < 1 or inv.get(act.item or "", 0) < act.qty):
                return "no_item"
            if k == "take" and (
                act.qty < 1 or obj.data["items"].get(act.item or "", 0) < act.qty
            ):
                return "chest_empty"
            if k == "sleep" and obj.owner not in (None, "shared", a.id):
                return "not_your_bed"
            if k == "trade":
                return self._check_trade(a, act)
            if k == "fulfill_order":
                order = self._order(act.item)
                if order is None or order.fulfilled_by is not None:
                    return "no_order"
                if inv.get(order.item, 0) < order.qty:
                    return "missing_items"
            return None
        if k == "eat":
            if act.item not in self._food:
                return "not_food"
            if inv.get(act.item, 0) < 1:
                return "no_item"
            return None
        if k == "give_item":
            b = self.actors.get(act.target_id or "")
            if b is None or b.id == a.id:
                return "no_target"
            if _cheb(a.pos, b.pos) > 2:
                return "too_far"
            if act.qty < 1 or inv.get(act.item or "", 0) < act.qty:
                return "no_item"
            return None
        if k == "move_to":
            if act.target_pos is None and act.target_id not in self.objects and (
                act.target_id not in self.actors
            ):
                return "no_target"
            return None
        if k in ("rest", "wait", "wake"):
            return None
        return "unknown_action"

    def _check_trade(self, a: ActorState, act: AtomicAction) -> str | None:
        if not self.merchant_present:
            return "merchant_absent"
        item = self.config.items.get(act.item or "")
        if item is None or act.qty < 1:
            return "bad_item"
        if act.op == "buy":
            if item.buy is None:
                return "not_sold"
            if a.inventory.get("coins", 0) < item.buy * act.qty:
                return "no_coins"
            return None
        if act.op == "sell":
            if item.sell is None:
                return "not_bought"
            if a.inventory.get(act.item or "", 0) < act.qty:
                return "no_item"
            return None
        return "bad_op"

    def _order(self, order_id: str | None) -> Order | None:
        return next((o for o in self.orders if o.id == order_id), None)

    def _duration(self, a: ActorState, act: AtomicAction) -> int:
        if act.kind in OBJECT_AFFORDANCES:
            obj = self.objects[act.target_id or ""]
            base = self.registry.get(obj.kind, act.kind).duration  # type: ignore[union-attr]
        elif act.kind == "rest":
            base = act.ticks or DEFAULT_REST_TICKS
        elif act.kind == "wait":
            base = act.ticks or DEFAULT_WAIT_TICKS
        else:
            base = ACTION_DURATIONS.get(act.kind, 1)
        if a.hunger >= self.config.needs.slow_hunger:
            base *= 2
        return base

    def _target_pos(self, act: AtomicAction) -> Tile | None:
        if act.target_id in self.objects:
            return self.objects[act.target_id].pos
        if act.target_id in self.actors:
            return self.actors[act.target_id].pos
        if act.target_pos is not None:
            return act.target_pos.as_tuple()
        return None

    def _start(self, a: ActorState, act: AtomicAction) -> ActionResult:
        k = act.kind
        events: list[WorldEvent] = []
        if k == "move_to":
            return self._start_move(a, act)
        if k == "wake":
            a.activity = "idle"
            events.append(self._event("woke", actor=a.id, pos=a.pos, importance=1))
            self._bump(a)
            return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick,
                                events=events)
        tpos = self._target_pos(act)
        if tpos is not None and tpos != a.pos:
            a.facing = facing_toward(a.pos, tpos, a.facing)
        if k == "sleep":
            a.activity = "sleeping"
            events.append(self._event("fell_asleep", actor=a.id, target=act.target_id,
                                      pos=a.pos, importance=1))
            self._bump(a)
            return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick,
                                events=events)
        if k == "give_item":  # instant; allowed during conversation
            events.extend(self._effect(a, act))
            self._bump(a)
            return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick,
                                events=events)
        duration = self._duration(a, act)
        a.action = ActiveAction(action=act, start_tick=self.tick, end_tick=self.tick + duration)
        a.activity = "resting" if k == "rest" else k
        events.append(self._event("action_started", actor=a.id, target=act.target_id, pos=a.pos,
                                  importance=0, action=k, end_tick=self.tick + duration))
        self._bump(a)
        return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick + duration,
                            events=events)

    def _start_move(self, a: ActorState, act: AtomicAction) -> ActionResult:
        goals: list[Tile]
        if act.target_id in self.objects or act.target_id in self.actors:
            tpos = self._target_pos(act)
            assert tpos is not None
            if _cheb(a.pos, tpos) <= 1:
                return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick)
            goals = self.grid.neighbors8(tpos)
        else:
            assert act.target_pos is not None
            near = self.grid.nearest_walkable(act.target_pos.as_tuple())
            goals = [near] if near else []
        path = self.grid.astar(a.pos, goals)
        if path is None:
            return self._reject(a.id, act, "no_path")
        if not path:
            return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=self.tick)
        step = 2 if a.hunger >= self.config.needs.slow_hunger else 1
        a.action = ActiveAction(action=act, start_tick=self.tick,
                                end_tick=self.tick + step * len(path), path=path,
                                next_step_tick=self.tick + step, step_ticks=step)
        a.activity = "walking"
        a.facing = facing_toward(a.pos, path[0], a.facing)
        self._bump(a)
        ev = self._event("path_move", actor=a.id, target=act.target_id, pos=a.pos, importance=0,
                         waypoints=[list(p) for p in path], step_ticks=step, facing=a.facing)
        return ActionResult(ok=True, actor_id=a.id, action=act, end_tick=a.action.end_tick,
                            events=[ev])

    # ------------------------------------------------------------------ effects
    def _gain(self, a: ActorState, yields: dict[str, int], bonus: dict[str, int]) -> dict[str, int]:
        got: dict[str, int] = {}
        for item, n in yields.items():
            got[item] = got.get(item, 0) + n
        for item, n in bonus.items():
            got[item] = got.get(item, 0) + self.rng.randint(0, n)
        for item, n in got.items():
            a.inventory[item] = a.inventory.get(item, 0) + n
        return got

    @staticmethod
    def _spend(inv: dict[str, int], item: str, n: int) -> None:
        inv[item] = inv.get(item, 0) - n
        if inv[item] <= 0:
            del inv[item]

    def _set_state(self, obj: ObjState, state: str) -> None:
        obj.state = state
        if "stage_days" in obj.data:
            obj.data["stage_days"] = 0
        self._bump(obj)

    def _effect(self, a: ActorState, act: AtomicAction) -> list[WorldEvent]:
        k = act.kind
        if k == "eat":
            item = self.config.items[act.item or ""]
            self._spend(a.inventory, act.item or "", 1)
            a.hunger = _clamp(a.hunger - item.food)
            a.stamina = _clamp(a.stamina + item.stamina)
            return [self._event("ate", actor=a.id, pos=a.pos, importance=1, item=act.item)]
        if k == "give_item":
            b = self.actors[act.target_id or ""]
            self._spend(a.inventory, act.item or "", act.qty)
            b.inventory[act.item or ""] = b.inventory.get(act.item or "", 0) + act.qty
            self._bump(b)
            return [self._event("gift", actor=a.id, target=b.id, pos=a.pos, importance=5,
                                item=act.item, qty=act.qty)]
        if k in ("rest", "wait"):
            return [self._event("rested" if k == "rest" else "waited", actor=a.id, pos=a.pos,
                                importance=0)]
        obj = self.objects[act.target_id or ""]
        aff = self.registry.get(obj.kind, k)
        assert aff is not None
        a.stamina = _clamp(a.stamina - aff.stamina_cost)
        evs: list[WorldEvent] = []
        if k == "plant":
            crop = self._crop_for_seed(act.item)
            self._spend(a.inventory, act.item or "", 1)
            obj.data = {"crop": crop, "growth": 0}
            self._set_state(obj, "watered" if self.weather == Weather.rain else "planted")
            evs.append(self._event("planted", actor=a.id, target=obj.id, pos=obj.pos, crop=crop))
        elif k == "harvest":
            crop_name = obj.data.get("crop", "turnip")
            crop = self.config.crops[crop_name]
            got = {crop_name: crop.yield_qty,
                   crop.seed: self.rng.randint(crop.seed_min, crop.seed_max)}
            for item, n in got.items():
                a.inventory[item] = a.inventory.get(item, 0) + n
            obj.data = {}
            self._set_state(obj, aff.to_state or "tilled")
            evs.append(self._event("harvested", actor=a.id, target=obj.id, pos=obj.pos,
                                   items=got, crop=crop_name))
        elif k == "cook":
            recipe = self.config.recipes[act.item or ""]
            for item, n in {**recipe.inputs}.items():
                self._spend(a.inventory, item, n)
            for item, n in recipe.fuel.items():
                self._spend(a.inventory, item, n)
            a.inventory[recipe.output] = a.inventory.get(recipe.output, 0) + 1
            evs.append(self._event("cooked", actor=a.id, target=obj.id, pos=obj.pos,
                                   recipe=act.item))
        elif k in ("store", "take"):
            evs.extend(self._chest(a, obj, act))
        elif k == "trade":
            item = self.config.items[act.item or ""]
            if act.op == "buy":
                self._spend(a.inventory, "coins", (item.buy or 0) * act.qty)
                a.inventory[act.item or ""] = a.inventory.get(act.item or "", 0) + act.qty
            else:
                self._spend(a.inventory, act.item or "", act.qty)
                a.inventory["coins"] = a.inventory.get("coins", 0) + (item.sell or 0) * act.qty
            evs.append(self._event("traded", actor=a.id, target=obj.id, pos=obj.pos,
                                   op=act.op, item=act.item, qty=act.qty))
        elif k == "fulfill_order":
            order = self._order(act.item)
            assert order is not None
            self._spend(a.inventory, order.item, order.qty)
            a.inventory["coins"] = a.inventory.get("coins", 0) + order.reward
            order.fulfilled_by = a.id
            self._bump(obj)
            evs.append(self._event("order_fulfilled", actor=a.id, target=obj.id, pos=obj.pos,
                                   importance=3, order=order.id, item=order.item,
                                   qty=order.qty, reward=order.reward))
        else:  # generic: chop / till / water / forage
            got = self._gain(a, aff.yields, aff.yields_bonus)
            if aff.to_state:
                self._set_state(obj, aff.to_state)
            evs.append(self._event(
                {"chop": "chopped", "till": "tilled", "water": "watered",
                 "forage": "foraged"}.get(k, k),
                actor=a.id, target=obj.id, pos=obj.pos, importance=2 if got else 1, items=got,
            ))
        return evs

    def _discover_missing(self, a: ActorState, chest: ObjState) -> list[WorldEvent]:
        missing = chest.data["missing"]
        chest.data["missing"] = []
        self._bump(chest)
        return [self._event("chest_missing_discovered", actor=a.id, target=a.id, pos=chest.pos,
                            importance=6, chest=chest.id,
                            missing=[{"item": m["item"], "qty": m["qty"]} for m in missing])]

    def _chest(self, a: ActorState, chest: ObjState, act: AtomicAction) -> list[WorldEvent]:
        items: dict[str, int] = chest.data["items"]
        item, qty = act.item or "", act.qty
        evs: list[WorldEvent] = []
        if chest.owner == a.id and chest.data.get("missing"):
            evs.extend(self._discover_missing(a, chest))
        if act.kind == "store":
            self._spend(a.inventory, item, qty)
            items[item] = items.get(item, 0) + qty
            evs.append(self._event("stored", actor=a.id, target=chest.id, pos=chest.pos,
                                   importance=1, item=item, qty=qty))
        else:
            self._spend(items, item, qty)
            a.inventory[item] = a.inventory.get(item, 0) + qty
            if chest.owner not in (None, "shared", a.id):
                witnesses = self._witnesses(a.pos, {a.id})
                chest.data["missing"].append(
                    {"taker": a.id, "item": item, "qty": qty, "tick": self.tick}
                )
                evs.append(self._event("chest_taken", actor=a.id, target=chest.owner,
                                       pos=chest.pos, importance=8, witnesses=witnesses,
                                       chest=chest.id, item=item, qty=qty))
            else:
                evs.append(self._event("took", actor=a.id, target=chest.id, pos=chest.pos,
                                       importance=1, item=item, qty=qty))
        self._bump(chest)
        return evs

    # ------------------------------------------------------------------ tick
    def tick_once(self) -> list[WorldEvent]:
        self.tick += 1
        events: list[WorldEvent] = []
        n = self.config.needs
        for a in self.actors.values():
            asleep = a.activity == "sleeping"
            a.hunger = _clamp(a.hunger + (n.hunger_per_min_asleep if asleep
                                          else n.hunger_per_min_awake))
            if asleep:
                a.stamina = _clamp(a.stamina + n.stamina_sleep_per_min)
            elif a.activity == "resting":
                a.stamina = _clamp(a.stamina + n.stamina_rest_per_min)
            target = 68.0 - max(0.0, a.hunger - 60.0) * 0.8 - (15.0 if a.stamina < 15 else 0.0)
            target -= {Weather.rain: 5.0, Weather.overcast: 2.0}.get(self.weather, 0.0)
            a.mood = _clamp(a.mood + (target - a.mood) * 0.01)
        for a in self.actors.values():
            act = a.action
            if act is None:
                continue
            if act.action.kind == "move_to":
                events.extend(self._step_move(a, act))
            elif self.tick >= act.end_tick:
                events.extend(self._complete(a))
        minute = self.minute
        if minute == 0:
            events.extend(self._daily_update())
        eco = self.config.economy
        if self.day % eco.merchant_every_days == 0:
            if minute == eco.merchant_open_minute:
                events.append(self._set_merchant(True))
            elif minute == eco.merchant_close_minute:
                events.append(self._set_merchant(False))
        return events

    def _set_merchant(self, present: bool) -> WorldEvent:
        """Toggle the traveling merchant; the stall's state/version change with it."""
        self.merchant_present = present
        stall = next((o for o in self.objects.values() if o.kind == "trading_stall"), None)
        if stall is not None:
            stall.state = "merchant_open" if present else "idle"
            self._bump(stall)
        else:
            self.world_version += 1
        return self._event("merchant_arrived" if present else "merchant_left",
                           target=stall.id if stall else None,
                           pos=stall.pos if stall else None, importance=3 if present else 1)

    def objects_of_kind_pos(self, kind: str) -> Tile | None:
        return next((o.pos for o in self.objects.values() if o.kind == kind), None)

    def _step_move(self, a: ActorState, act: ActiveAction) -> list[WorldEvent]:
        if self.tick < act.next_step_tick:
            return []
        nxt = act.path[0]
        if not self.grid.walkable(nxt):
            a.action = None
            a.activity = "idle"
            self._bump(a)
            return [self._event("action_failed", actor=a.id, pos=a.pos, importance=1,
                                action="move_to", reason="blocked")]
        a.facing = facing_toward(a.pos, nxt, a.facing)
        a.pos = nxt
        act.path.pop(0)
        act.next_step_tick += act.step_ticks
        self._bump(a)
        if act.path:
            return []
        a.action = None
        a.activity = "idle"
        tpos = self._target_pos(act.action)
        if tpos is not None and tpos != a.pos and act.action.target_id:
            a.facing = facing_toward(a.pos, tpos, a.facing)
        return [self._event("action_completed", actor=a.id, target=act.action.target_id,
                            pos=a.pos, importance=0, action="move_to")]

    def _complete(self, a: ActorState) -> list[WorldEvent]:
        assert a.action is not None
        act = a.action.action
        a.action = None
        a.activity = "idle"
        reason = self._check(a, act)
        if reason:
            self._bump(a)
            return [self._event("action_failed", actor=a.id, target=act.target_id, pos=a.pos,
                                importance=1, action=act.kind, reason=reason)]
        evs = self._effect(a, act)
        self._bump(a)
        evs.append(self._event("action_completed", actor=a.id, target=act.target_id, pos=a.pos,
                               importance=0, action=act.kind))
        return evs

    def _daily_update(self) -> list[WorldEvent]:
        evs: list[WorldEvent] = [self._event("new_day", importance=0, day=self.day)]
        rained = self.weather == Weather.rain
        for o in self.objects.values():
            if o.kind == "farm_plot" and o.state in ("planted", "watered"):
                if o.state == "watered" or rained:
                    o.data["growth"] = o.data.get("growth", 0) + 1
                crop = self.config.crops[o.data.get("crop", "turnip")]
                o.state = "ready" if o.data.get("growth", 0) >= crop.days else "planted"
                self._bump(o)
                continue
            growth = self.config.kinds[o.kind].growth
            if growth is None or o.state == growth.stages[-1] or o.state not in growth.stages:
                continue
            idx = growth.stages.index(o.state)
            o.data["stage_days"] = o.data.get("stage_days", 0) + 1
            if o.data["stage_days"] >= growth.days[idx]:
                o.state = growth.stages[idx + 1]
                o.data["stage_days"] = 0
                self._bump(o)
        weights = self.config.economy.weather_weights
        names = list(weights)
        self.weather = Weather(self.rng.choices(names, weights=[weights[n] for n in names])[0])
        if self.weather == Weather.rain:
            self._rain_waters()
        evs.append(self._event("weather", importance=1, weather=self.weather.value))
        evs.extend(self._post_orders())
        return evs

    def _rain_waters(self) -> None:
        for o in self.objects.values():
            if o.kind == "farm_plot" and o.state == "planted":
                o.state = "watered"
                self._bump(o)

    def _post_orders(self) -> list[WorldEvent]:
        eco = self.config.economy
        day = self.day
        self.orders = [o for o in self.orders if o.fulfilled_by is None and o.expires_day >= day]
        evs: list[WorldEvent] = []
        board = self.objects_of_kind_pos("notice_board")
        for _ in range(eco.orders_per_day):
            if not eco.order_templates:
                break
            t = self.rng.choice(eco.order_templates)
            qty = self.rng.randint(t.min_qty, t.max_qty)
            self.order_seq += 1
            order = Order(id=f"order_{self.order_seq:04d}", item=t.item, qty=qty,
                          reward=qty * t.reward_per_unit, posted_day=day,
                          expires_day=day + eco.order_lifetime_days)
            self.orders.append(order)
            evs.append(self._event("order_posted", pos=board, importance=1, order=order.id,
                                   item=order.item, qty=qty, reward=order.reward))
        self.world_version += 1
        return evs

    # ------------------------------------------------------------------ engine / God hooks
    def interrupt(self, actor_id: str, reason: str) -> list[WorldEvent]:
        a = self.actors[actor_id]
        if a.action is None:
            return []
        kind = a.action.action.kind
        a.action = None
        a.activity = "idle"
        self._bump(a)
        return [self._event("action_interrupted", actor=a.id, pos=a.pos, importance=1,
                            action=kind, reason=reason)]

    def set_activity(self, actor_id: str, activity: str) -> None:
        a = self.actors[actor_id]
        a.activity = activity
        self._bump(a)

    def face_toward(self, actor_id: str, pos: Tile) -> None:
        a = self.actors[actor_id]
        a.facing = facing_toward(a.pos, pos, a.facing)
        self._bump(a)

    def adjust_mood(self, actor_id: str, delta: float) -> None:
        a = self.actors[actor_id]
        a.mood = _clamp(a.mood + delta)

    def set_weather(self, weather: Weather) -> list[WorldEvent]:
        self.weather = Weather(weather)
        if self.weather == Weather.rain:
            self._rain_waters()
        self.world_version += 1
        return [self._event("weather_changed", importance=4, weather=self.weather.value)]

    def spawn_resource(self, kind: str, pos: Tile, state: str | None = None) -> list[WorldEvent]:
        kdef = self.config.kinds.get(kind)
        if kdef is None:
            raise ValueError(f"unknown kind {kind!r}")
        if not self.grid.walkable(pos) or any(a.pos == pos for a in self.actors.values()):
            raise ValueError(f"tile {pos} is not free")
        self.spawn_seq += 1
        oid = f"{kind}_spawn_{self.spawn_seq:03d}"
        data: dict[str, Any] = {"stage_days": 0} if kdef.growth else {}
        if kind == "chest":
            data = {"items": {}, "missing": []}
        obj = ObjState(id=oid, kind=kind, pos=pos, state=state or kdef.initial_state,
                       owner="shared" if kind == "chest" else None, name=oid, data=data)
        self.objects[oid] = obj
        self._bump(obj)
        self._rebuild_static()
        return [self._event("resource_spawned", target=oid, pos=pos, importance=5,
                            object_kind=kind)]

    # ------------------------------------------------------------------ stats / persistence
    def stats(self) -> dict[str, int]:
        objs = list(self.objects.values())
        holders = [a.inventory for a in self.actors.values()] + [
            o.data["items"] for o in objs if o.kind == "chest"
        ]
        seed_names = {c.seed for c in self.config.crops.values()}

        def total(names: set[str] | frozenset[str]) -> int:
            return sum(n for inv in holders for k, n in inv.items() if k in names)

        return {
            "trees_total": sum(o.kind == "tree" for o in objs),
            "trees_mature": sum(o.kind == "tree" and o.state == "mature" for o in objs),
            "bushes_total": sum(o.kind == "berry_bush" for o in objs),
            "bushes_ripe": sum(o.kind == "berry_bush" and o.state == "ripe" for o in objs),
            "plots_growing": sum(
                o.kind == "farm_plot" and o.state in ("planted", "watered", "ready") for o in objs
            ),
            "seeds_total": total(seed_names),
            "food_total": total(self._food),
            "wood_total": total({"wood"}),
            "coins_total": total({"coins"}),
        }

    def snapshot(self) -> dict[str, Any]:
        state = self.rng.getstate()
        return {
            "tick": self.tick,
            "world_version": self.world_version,
            "weather": self.weather.value,
            "merchant_present": self.merchant_present,
            "order_seq": self.order_seq,
            "spawn_seq": self.spawn_seq,
            "rng": [state[0], list(state[1]), state[2]],
            "objects": [o.model_dump(mode="json") for o in self.objects.values()],
            "actors": [a.model_dump(mode="json") for a in self.actors.values()],
            "orders": [o.model_dump(mode="json") for o in self.orders],
        }

    @classmethod
    def restore(cls, config: VillageConfig, data: dict[str, Any]) -> World:
        w = cls(config)
        w.tick = data["tick"]
        w.world_version = data["world_version"]
        w.weather = Weather(data["weather"])
        w.merchant_present = data["merchant_present"]
        w.order_seq = data["order_seq"]
        w.spawn_seq = data["spawn_seq"]
        r = data["rng"]
        w.rng.setstate((r[0], tuple(r[1]), r[2]))
        w.objects = {o["id"]: ObjState.model_validate(o) for o in data["objects"]}
        w.actors = {a["id"]: ActorState.model_validate(a) for a in data["actors"]}
        w.orders = [Order.model_validate(o) for o in data["orders"]]
        w._rebuild_static()
        return w

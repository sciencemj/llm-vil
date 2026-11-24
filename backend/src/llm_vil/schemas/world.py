from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .base import Frozen


class TilePos(Frozen):
    x: int
    y: int

    def chebyshev(self, other: TilePos) -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


Facing = int
"""Facing in degrees, snapped to 45° steps. 0 = east (+x), 90 = south (+y)."""


class Weather(StrEnum):
    sunny = "sunny"
    overcast = "overcast"
    rain = "rain"


class NeedsState(Frozen):
    hunger: float = Field(ge=0, le=100)
    stamina: float = Field(ge=0, le=100)
    mood: float = Field(ge=0, le=100)
    hp: float = Field(default=100.0, ge=0, le=100)


Inventory = dict[str, int]


class WorldObjectState(Frozen):
    id: str
    kind: str
    pos: TilePos
    state: str
    owner: str | None = None
    name: str = ""
    version: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class ActorSnapshot(Frozen):
    id: str
    name: str
    pos: TilePos
    facing: Facing
    activity: str
    needs: NeedsState
    inventory: Inventory = Field(default_factory=dict)
    version: int = 0
    profession: str = ""
    skills: list[str] = Field(default_factory=list)
    # slot -> {"item": str, "durability": int, "max_durability": int}
    equipment: dict[str, dict[str, Any]] = Field(default_factory=dict)


class OrderState(Frozen):
    id: str
    item: str
    qty: int
    reward: int
    posted_day: int
    expires_day: int
    fulfilled_by: str | None = None
    kind: str = "order"          # order (kingdom/villager request) | bounty (guild hunt)
    poster_id: str | None = None  # None = the kingdom's daily orders
    escrow: int = 0               # coins held until fulfilment / refunded on expiry


class WorldView(Frozen):
    """What one actor perceives at one tick. Stamped with ``world_version``."""

    world_version: int
    tick: int
    day: int
    minute: int
    weather: Weather
    self_actor: ActorSnapshot
    objects: list[WorldObjectState]
    actors: list[ActorSnapshot]
    orders: list[OrderState]
    merchant_present: bool


AtomicKind = Literal[
    "move_to",
    "chop",
    "till",
    "plant",
    "water",
    "harvest",
    "forage",
    "cook",
    "eat",
    "sleep",
    "wake",
    "rest",
    "wait",
    "store",
    "take",
    "give_item",
    "trade",
    "fulfill_order",
    "talk",  # engine-level intent: opens a ConversationSession, never reaches World.apply
    # Phase 2: professions, shops, guild, combat
    "craft",
    "mine",
    "gather_herb",
    "fight",
    "stock_shop",
    "buy_from_shop",
    "set_prices",
    "sell_drops",
    "post_request",
    "post_bounty",
    "repair_tool",
]


class AtomicAction(Frozen):
    kind: AtomicKind
    target_id: str | None = None
    target_pos: TilePos | None = None
    item: str | None = None
    qty: int = 1
    op: str | None = None
    ticks: int = 0
    amount: int = 0  # coins: request/bounty reward or shop price


class ActionIntent(Frozen):
    actor_id: str
    action: AtomicAction
    view_version: int
    source: Literal["agent", "human", "engine"] = "agent"


class WorldEvent(Frozen):
    tick: int
    kind: str
    actor_id: str | None = None
    target_id: str | None = None
    pos: TilePos | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    importance: int = 2
    witnesses: list[str] = Field(default_factory=list)


class ActionResult(Frozen):
    ok: bool
    actor_id: str
    action: AtomicAction
    reason: str = ""
    end_tick: int = 0
    events: list[WorldEvent] = Field(default_factory=list)

from __future__ import annotations

from typing import Any

from pydantic import Field

from .agent import AgentSpec
from .base import Frozen


class AffordanceDef(Frozen):
    from_states: list[str] = Field(default_factory=list)  # empty = any state
    to_state: str | None = None
    stamina_cost: float = 0.0
    duration: int = 1
    reach: int = 1
    yields: dict[str, int] = Field(default_factory=dict)
    yields_bonus: dict[str, int] = Field(default_factory=dict)  # extra 0..n (seeded RNG)


class GrowthDef(Frozen):
    stages: list[str]
    days: list[int]


class ObjectKindDef(Frozen):
    initial_state: str
    blocking: bool = True
    opaque: bool = False
    affordances: dict[str, AffordanceDef] = Field(default_factory=dict)
    growth: GrowthDef | None = None


class ItemDef(Frozen):
    food: float = 0.0
    stamina: float = 0.0
    buy: int | None = None
    sell: int | None = None


class CropDef(Frozen):
    seed: str
    days: int
    yield_qty: int
    seed_min: int = 1
    seed_max: int = 2


class RecipeDef(Frozen):
    inputs: dict[str, int]
    fuel: dict[str, int] = Field(default_factory=dict)
    output: str


class ObjectSpawn(Frozen):
    id: str
    kind: str
    pos: tuple[int, int]
    state: str | None = None
    owner: str | None = None
    name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AreaDef(Frozen):
    id: str
    name: str
    rect: tuple[int, int, int, int]  # x, y, w, h

    def contains(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def center(self) -> tuple[int, int]:
        rx, ry, rw, rh = self.rect
        return (rx + rw // 2, ry + rh // 2)


class ClockConfig(Frozen):
    sim_minutes_per_real_second: float = 2.0
    t_max_seconds: float = 3.0


class NeedsConfig(Frozen):
    hunger_per_min_awake: float = 0.06
    hunger_per_min_asleep: float = 0.02
    stamina_sleep_per_min: float = 0.25
    stamina_rest_per_min: float = 0.2
    hungry_threshold: float = 50
    tired_threshold: float = 25
    slow_hunger: float = 85


class OrderTemplate(Frozen):
    item: str
    min_qty: int
    max_qty: int
    reward_per_unit: int


class EconomyConfig(Frozen):
    forest_reserve: int = 3
    orders_per_day: int = 2
    order_templates: list[OrderTemplate] = Field(default_factory=list)
    order_lifetime_days: int = 2
    merchant_every_days: int = 3
    merchant_open_minute: int = 480
    merchant_close_minute: int = 1080
    weather_weights: dict[str, float] = Field(default_factory=lambda: {"sunny": 1.0})


class MemoryConfig(Frozen):
    hot_days: int = 3
    salient_importance: int = 7
    reflection_threshold: int = 120
    perception_cooldown: int = 60
    retrieve_k: int = 6


class ConversationConfig(Frozen):
    max_turns: int = 6
    ticks_per_turn: int = 2


class VillageConfig(Frozen):
    name: str
    language: str = "ko"
    seed: int = 0
    width: int
    height: int
    ticks_per_day: int = 1440
    start_minute: int = 300
    perception_radius: int = 6
    clock: ClockConfig = ClockConfig()
    needs: NeedsConfig = NeedsConfig()
    economy: EconomyConfig = EconomyConfig()
    memory: MemoryConfig = MemoryConfig()
    conversation: ConversationConfig = ConversationConfig()
    items: dict[str, ItemDef]
    crops: dict[str, CropDef]
    recipes: dict[str, RecipeDef]
    kinds: dict[str, ObjectKindDef]
    water: list[tuple[int, int, int, int]] = Field(default_factory=list)
    areas: list[AreaDef]
    objects: list[ObjectSpawn]
    villagers: list[AgentSpec]

    def area(self, area_id: str) -> AreaDef | None:
        return next((a for a in self.areas if a.id == area_id), None)

    def area_of(self, x: int, y: int) -> str | None:
        return next((a.id for a in self.areas if a.contains(x, y)), None)

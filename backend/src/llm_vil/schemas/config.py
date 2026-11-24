from __future__ import annotations

from typing import Any

from pydantic import Field

from .agent import AgentSpec
from .base import Frozen


class AffordanceDef(Frozen):
    from_states: list[str] = Field(default_factory=list)  # empty = any state
    requires_skill: str | None = None  # hard gate: exclusive profession stations
    requires_tool: str | None = None   # soft gate: without it the work is slower / smaller
    to_state: str | None = None
    stamina_cost: float = 0.0
    duration: int = 1
    reach: int = 1
    yields: dict[str, int] = Field(default_factory=dict)
    yields_bonus: dict[str, int] = Field(default_factory=dict)  # extra 0..n (seeded RNG)
    yields_no_tool: dict[str, int] | None = None  # replaces yields when the tool is missing


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
    value: int = 1  # base price; shops may charge 0.7-1.5x
    hp: float = 0.0  # potions heal
    equip_slot: str | None = None  # tool | weapon | armor
    max_durability: int = 0
    drop: bool = False  # monster drop (the guild buys these)
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
    station: str = "hearth"  # object kind where it is crafted
    skill: str | None = None  # profession skill required (None = anyone)
    output_qty: int = 1
    fuel_any: list[str] = Field(default_factory=list)  # one unit of the first one held
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


class MonsterDef(Frozen):
    power: float
    drops: dict[str, int] = Field(default_factory=dict)
    drops_bonus: dict[str, int] = Field(default_factory=dict)
    respawn_minutes: int = 360
    damage: float = 15.0


class CombatConfig(Frozen):
    base_win: float = 0.45
    weapon_bonus: float = 0.3
    armor_bonus: float = 0.15
    min_win: float = 0.15
    max_win: float = 0.95
    lose_damage: float = 35.0
    faint_hp: float = 20.0
    recover_hp: float = 45.0
    hp_regen_awake: float = 0.03
    hp_regen_asleep: float = 0.12
    roam_every: int = 7
    zone: tuple[int, int, int, int] = (33, 2, 7, 26)  # x, y, w, h monsters never leave
    clinic: str = "tavern_main"  # where fainted villagers wake up


class ShopConfig(Frozen):
    price_min: float = 0.7
    price_max: float = 1.5
    repair_fee: int = 6
    guild_stipend: int = 60
    guild_treasury_floor: int = 80
    guild_buy_ratio: float = 0.8
    guild_stock_cap: int = 10       # the guild stops buying a drop it already has this many of
    tax_threshold: int = 100        # daily village tax on savings above this ...
    tax_rate: float = 0.1           # ... at this rate; half funds the guild, half is burned
    guild_wage: int = 15            # daily wage the guild treasury pays its guildmaster


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
    walls: list[tuple[int, int, int, int]] = Field(default_factory=list)  # fences, impassable
    monsters: dict[str, MonsterDef] = Field(default_factory=dict)
    combat: CombatConfig = CombatConfig()
    shops: ShopConfig = ShopConfig()
    areas: list[AreaDef]
    objects: list[ObjectSpawn]
    villagers: list[AgentSpec]

    def area(self, area_id: str) -> AreaDef | None:
        return next((a for a in self.areas if a.id == area_id), None)

    def area_of(self, x: int, y: int) -> str | None:
        return next((a.id for a in self.areas if a.contains(x, y)), None)

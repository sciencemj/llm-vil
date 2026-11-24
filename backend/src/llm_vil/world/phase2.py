"""Phase 2 world mechanics: professions, crafting stations, tools & durability, P2P shops,
the adventurers' guild, villager requests / bounties, and the Mosswood monster loop.

Mixed into :class:`World`; every effect still goes through ``World.apply`` (validity gate)
and ``World.tick_once``. Never imports ``llm_vil.agent``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..schemas.config import RecipeDef
from ..schemas.world import AtomicAction, WorldEvent

if TYPE_CHECKING:  # pragma: no cover
    from .world import ActorState, ObjState, World

SHOP_KINDS = frozenset({"shop_counter", "guild_counter"})
CRAFT_KINDS = frozenset({"cook", "craft"})
MAX_OPEN_BOUNTIES = 3
MAX_OPEN_REQUESTS_PER_VILLAGER = 2


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Phase2Mixin:
    """Methods expect to be bound to a ``World`` (see TYPE_CHECKING import)."""

    # ------------------------------------------------------------------ setup
    def _init_phase2(self: World) -> None:
        specs = {v.id: v for v in self.config.villagers}
        for a in self.actors.values():
            spec = specs.get(a.id)
            if spec is not None:
                a.profession = spec.profession
                a.skills = list(spec.skills)
            self._auto_equip(a)
        for o in self.objects.values():
            if o.kind in SHOP_KINDS:
                o.data.setdefault("stock", {})
                o.data.setdefault("prices", {})
                o.data.setdefault("open", True)
            if o.kind == "guild_counter":
                o.data.setdefault("treasury", self.config.shops.guild_treasury_floor)
            if o.kind == "monster":
                o.data.setdefault("home", list(o.pos))
                o.data.setdefault("respawn_tick", 0)

    # ------------------------------------------------------------------ items / tools
    def value_of(self: World, item: str) -> int:
        it = self.config.items.get(item)
        return max(1, it.value) if it else 1

    def price_band(self: World, item: str) -> tuple[int, int]:
        v = self.value_of(item)
        s = self.config.shops
        return max(1, math.floor(v * s.price_min)), max(1, math.ceil(v * s.price_max))

    def shop_price(self: World, shop: ObjState, item: str) -> int:
        return int(shop.data.get("prices", {}).get(item) or self.value_of(item))

    def _auto_equip(self: World, a: ActorState) -> None:
        """Equip the best held item for each empty or broken slot (one unit per slot)."""
        for item, n in sorted(a.inventory.items()):
            if n <= 0:
                continue
            it = self.config.items.get(item)
            if it is None or not it.equip_slot:
                continue
            cur = a.equipment.get(it.equip_slot)
            if cur is None or cur.get("durability", 0) <= 0:
                self._spend(a.inventory, item, 1)
                a.equipment[it.equip_slot] = {"item": item, "durability": it.max_durability,
                                              "max_durability": it.max_durability}

    def _has_tool(self: World, a: ActorState, tool: str | None) -> bool:
        if not tool:
            return True
        for eq in a.equipment.values():
            if eq.get("item") == tool and eq.get("durability", 0) > 0:
                return True
        return False

    def _wear(self: World, a: ActorState, tool: str | None) -> None:
        for eq in a.equipment.values():
            if eq.get("item") == tool and eq.get("durability", 0) > 0:
                eq["durability"] -= 1
                return

    # ------------------------------------------------------------------ recipes
    def recipe_needs(self: World, recipe: RecipeDef, inv: dict[str, int]) -> dict[str, int] | None:
        """Inputs + fuel this recipe would consume from ``inv``, or None if it can't."""
        need = dict(recipe.inputs)
        for item, n in recipe.fuel.items():
            need[item] = need.get(item, 0) + n
        if recipe.fuel_any:
            fuel = next((f for f in recipe.fuel_any if inv.get(f, 0) >= need.get(f, 0) + 1), None)
            if fuel is None:
                return None
            need[fuel] = need.get(fuel, 0) + 1
        if any(inv.get(i, 0) < n for i, n in need.items()):
            return None
        return need

    # ------------------------------------------------------------------ validity
    def _check_phase2(self: World, a: ActorState, act: AtomicAction, obj: ObjState) -> str | None:
        k = act.kind
        inv = a.inventory
        if k in CRAFT_KINDS:
            recipe = self.config.recipes.get(act.item or "")
            if recipe is None:
                return "bad_recipe"
            if recipe.station != obj.kind:
                return "wrong_station"
            if recipe.skill and recipe.skill not in a.skills:
                return "needs_skill"
            if self.recipe_needs(recipe, inv) is None:
                return "missing_ingredients"
            return None
        if k == "stock_shop":
            if obj.owner != a.id:
                return "not_your_shop"
            if act.item == "coins":
                return "bad_item"
            if act.qty < 1 or inv.get(act.item or "", 0) < act.qty:
                return "no_item"
            return None
        if k == "buy_from_shop":
            if obj.owner == a.id:
                return "own_shop"
            if act.qty < 1 or obj.data["stock"].get(act.item or "", 0) < act.qty:
                return "out_of_stock"
            if inv.get("coins", 0) < self.shop_price(obj, act.item or "") * act.qty:
                return "no_coins"
            return None
        if k == "set_prices":
            if obj.owner != a.id:
                return "not_your_shop"
            lo, hi = self.price_band(act.item or "")
            if not lo <= act.amount <= hi:
                return f"price_out_of_band_{lo}_{hi}"
            return None
        if k == "sell_drops":
            it = self.config.items.get(act.item or "")
            if it is None or not it.drop:
                return "not_a_drop"
            if act.qty < 1 or inv.get(act.item or "", 0) < act.qty:
                return "no_item"
            cur_stock = obj.data["stock"].get(act.item or "", 0)
            if cur_stock + act.qty > self.config.shops.guild_stock_cap:
                return "guild_full"
            if obj.data.get("treasury", 0) < self._drop_price(act.item or "") * act.qty:
                return "treasury_empty"
            return None
        if k == "post_bounty":
            if obj.owner != a.id:
                return "not_guildmaster"
            if (act.item or "") not in self.config.monsters:
                return "bad_species"
            if act.amount < 1 or obj.data.get("treasury", 0) < act.amount:
                return "treasury_empty"
            open_b = sum(1 for o in self.orders if o.kind == "bounty" and not o.fulfilled_by)
            if open_b >= MAX_OPEN_BOUNTIES:
                return "too_many_bounties"
            return None
        if k == "post_request":
            if (act.item or "") not in self.config.items or act.item == "coins":
                return "bad_item"
            if act.qty < 1 or act.amount < 1 or inv.get("coins", 0) < act.amount:
                return "no_coins"
            mine = sum(1 for o in self.orders
                       if o.kind == "order" and o.poster_id == a.id and not o.fulfilled_by)
            if mine >= MAX_OPEN_REQUESTS_PER_VILLAGER:
                return "too_many_requests"
            return None
        if k == "repair_tool":
            eq = self._equipment_slot(a, act.item)
            if eq is None:
                return "nothing_to_repair"
            if eq["durability"] >= eq["max_durability"]:
                return "not_worn"
            if "smithing" not in a.skills and inv.get("coins", 0) < self.config.shops.repair_fee:
                return "no_coins"
            return None
        if k == "fight":
            engaged_by = obj.data.get("engaged_by")
            if obj.state not in ("roaming",) and not (obj.state == "engaged"
                                                     and engaged_by == a.id):
                return f"state_{obj.state}"
            if a.hp < self.config.combat.faint_hp:
                return "too_hurt"
            return None
        return None

    def _equipment_slot(self: World, a: ActorState, key: str | None) -> dict[str, Any] | None:
        if key in a.equipment:
            return a.equipment[key]
        return next((eq for eq in a.equipment.values() if eq.get("item") == key), None)

    def _drop_price(self: World, item: str) -> int:
        return max(1, round(self.value_of(item) * self.config.shops.guild_buy_ratio))

    # ------------------------------------------------------------------ effects
    def _effect_phase2(self: World, a: ActorState, act: AtomicAction,
                       obj: ObjState) -> list[WorldEvent]:
        k = act.kind
        inv = a.inventory
        if k in CRAFT_KINDS:
            recipe = self.config.recipes[act.item or ""]
            need = self.recipe_needs(recipe, inv) or {}
            for item, n in need.items():
                self._spend(inv, item, n)
            inv[recipe.output] = inv.get(recipe.output, 0) + recipe.output_qty
            self._auto_equip(a)
            evs = [self._event("crafted", actor=a.id, target=obj.id, pos=obj.pos, importance=2,
                               recipe=act.item, station=obj.kind, output=recipe.output,
                               qty=recipe.output_qty)]
            if obj.kind == "hearth":  # keep the Phase 1 "cooked" signal for meals
                evs.append(self._event("cooked", actor=a.id, target=obj.id, pos=obj.pos,
                                       recipe=act.item))
            return evs
        if k == "stock_shop":
            stock = obj.data["stock"]
            self._spend(inv, act.item or "", act.qty)
            stock[act.item or ""] = stock.get(act.item or "", 0) + act.qty
            obj.data["prices"].setdefault(act.item or "", self.value_of(act.item or ""))
            self._bump(obj)
            return [self._event("shop_stocked", actor=a.id, target=obj.id, pos=obj.pos,
                                importance=1, item=act.item, qty=act.qty)]
        if k == "buy_from_shop":
            item, qty = act.item or "", act.qty
            price = self.shop_price(obj, item)
            self._spend(obj.data["stock"], item, qty)
            self._spend(inv, "coins", price * qty)
            self._pay(obj, price * qty)
            inv[item] = inv.get(item, 0) + qty
            self._auto_equip(a)
            self._bump(obj)
            return [self._event("shop_sale", actor=a.id, target=obj.owner, pos=obj.pos,
                                importance=2, shop=obj.id, item=item, qty=qty, price=price,
                                total=price * qty)]
        if k == "set_prices":
            obj.data["prices"][act.item or ""] = act.amount
            self._bump(obj)
            return [self._event("price_set", actor=a.id, target=obj.id, pos=obj.pos,
                                importance=1, item=act.item, price=act.amount)]
        if k == "sell_drops":
            item, qty = act.item or "", act.qty
            pay = self._drop_price(item) * qty
            obj.data["treasury"] -= pay
            self._spend(inv, item, qty)
            inv["coins"] = inv.get("coins", 0) + pay
            obj.data["stock"][item] = obj.data["stock"].get(item, 0) + qty
            obj.data["prices"].setdefault(item, self.value_of(item))
            self._bump(obj)
            return [self._event("drops_sold", actor=a.id, target=obj.owner, pos=obj.pos,
                                importance=3, item=item, qty=qty, coins=pay)]
        if k == "post_bounty":
            obj.data["treasury"] -= act.amount
            self._bump(obj)
            return [self._new_order(a.id, act.item or "", 1, act.amount, "bounty", obj)]
        if k == "post_request":
            self._spend(inv, "coins", act.amount)
            return [self._new_order(a.id, act.item or "", act.qty, act.amount, "order", obj)]
        if k == "repair_tool":
            eq = self._equipment_slot(a, act.item)
            assert eq is not None
            fee = 0 if "smithing" in a.skills else self.config.shops.repair_fee
            if fee:
                self._spend(inv, "coins", fee)
                self._pay(obj, fee)
            eq["durability"] = eq["max_durability"]
            return [self._event("tool_repaired", actor=a.id, target=obj.owner, pos=obj.pos,
                                importance=2, item=eq["item"], fee=fee)]
        if k == "fight":
            return self._resolve_fight(a, obj)
        return []

    def _pay(self: World, shop: ObjState, coins: int) -> None:
        """Coins paid at a counter go to its owner (the guild's go to its treasury)."""
        if shop.kind == "guild_counter":
            shop.data["treasury"] = shop.data.get("treasury", 0) + coins
            return
        owner = self.actors.get(shop.owner or "")
        if owner is not None:
            owner.inventory["coins"] = owner.inventory.get("coins", 0) + coins
            self._bump(owner)

    def _new_order(self: World, poster: str, item: str, qty: int, reward: int, kind: str,
                   at: ObjState) -> WorldEvent:
        from .world import Order

        self.order_seq += 1
        order = Order(id=f"{kind}_{self.order_seq:04d}", item=item, qty=qty, reward=reward,
                      posted_day=self.day,
                      expires_day=self.day + self.config.economy.order_lifetime_days + 1,
                      kind=kind, poster_id=poster, escrow=reward)
        self.orders.append(order)
        self.world_version += 1
        return self._event("bounty_posted" if kind == "bounty" else "request_posted",
                           actor=poster, target=at.id, pos=at.pos, importance=3,
                           order=order.id, item=item, qty=qty, reward=reward)

    def _fulfill_villager_order(self: World, a: ActorState, order: Any) -> None:
        """(request) items go straight to the poster; the escrow pays the fulfiller."""
        poster = self.actors.get(order.poster_id or "")
        if poster is not None:
            poster.inventory[order.item] = poster.inventory.get(order.item, 0) + order.qty
            self._auto_equip(poster)
            self._bump(poster)
        a.inventory["coins"] = a.inventory.get("coins", 0) + order.escrow
        order.escrow = 0

    # ------------------------------------------------------------------ combat
    def _resolve_fight(self: World, a: ActorState, mon: ObjState) -> list[WorldEvent]:
        c = self.config.combat
        spec = self.config.monsters[mon.data["species"]]
        p = c.base_win - spec.power
        if self._has_tool(a, "sword"):
            p += c.weapon_bonus
        if a.equipment.get("armor", {}).get("durability", 0) > 0:
            p += c.armor_bonus
        p = min(c.max_win, max(c.min_win, p))
        won = self.rng.random() < p
        mon.data.pop("engaged_by", None)
        evs: list[WorldEvent] = []
        if won:
            got = self._gain(a, spec.drops, spec.drops_bonus)
            self._wear(a, "sword")
            armor = a.equipment.get("armor")
            has_armor = armor is not None and armor.get("durability", 0) > 0
            if has_armor:
                assert armor is not None
                armor["durability"] -= 1
            a.hp = max(0.0, a.hp - spec.damage * (0.5 if has_armor else 1.0))
            mon.state = "defeated"
            mon.data["respawn_tick"] = self.tick + spec.respawn_minutes
            self._bump(mon)
            evs.append(self._event("monster_defeated", actor=a.id, target=mon.id, pos=mon.pos,
                                   importance=4, species=mon.data["species"], items=got,
                                   p_win=round(p, 3)))
            evs.extend(self._complete_bounty(a, mon.data["species"], mon))
        else:
            a.hp = max(0.0, a.hp - c.lose_damage)
            mon.state = "roaming"
            self._bump(mon)
            evs.append(self._event("fight_lost", actor=a.id, target=mon.id, pos=mon.pos,
                                   importance=5, species=mon.data["species"], p_win=round(p, 3)))
        if a.hp < c.faint_hp:
            evs.extend(self._faint(a))
        return evs

    def _complete_bounty(self: World, a: ActorState, species: str,
                         mon: ObjState) -> list[WorldEvent]:
        bounty = next((o for o in self.orders if o.kind == "bounty" and o.item == species
                       and o.fulfilled_by is None), None)
        if bounty is None:
            return []
        a.inventory["coins"] = a.inventory.get("coins", 0) + bounty.escrow
        reward, bounty.escrow = bounty.escrow, 0
        bounty.fulfilled_by = a.id
        self.world_version += 1
        return [self._event("bounty_completed", actor=a.id, target=bounty.poster_id,
                            pos=mon.pos, importance=4, order=bounty.id, species=species,
                            reward=reward)]

    def _faint(self: World, a: ActorState) -> list[WorldEvent]:
        """Non-lethal: lose half the unbanked monster drops, wake up at the clinic."""
        c = self.config.combat
        lost: dict[str, int] = {}
        for item, n in sorted(a.inventory.items()):
            it = self.config.items.get(item)
            if it is not None and it.drop and n > 0:
                drop = math.ceil(n / 2)
                self._spend(a.inventory, item, drop)
                lost[item] = drop
        clinic = self.objects.get(c.clinic)
        if clinic is not None:
            spots = [t for t in self.grid.neighbors8(clinic.pos)]
            if spots:
                a.pos = sorted(spots)[0]
        a.action = None
        a.activity = "idle"
        a.hp = c.recover_hp
        a.stamina = min(a.stamina, 40.0)
        self._bump(a)
        return [self._event("fainted", actor=a.id, target=a.id, pos=a.pos, importance=7,
                            lost=lost, clinic=c.clinic)]

    def release_monster(self: World, actor_id: str) -> None:
        for o in self.objects.values():
            if o.kind == "monster" and o.data.get("engaged_by") == actor_id:
                o.data.pop("engaged_by", None)
                if o.state == "engaged":
                    o.state = "roaming"
                self._bump(o)

    # ------------------------------------------------------------------ per tick / day
    def _tick_phase2(self: World) -> list[WorldEvent]:
        c = self.config.combat
        evs: list[WorldEvent] = []
        for a in self.actors.values():
            regen = c.hp_regen_asleep if a.activity == "sleeping" else c.hp_regen_awake
            if a.hp < 100.0:
                a.hp = min(100.0, a.hp + regen)
        zx, zy, zw, zh = c.zone
        for o in self.objects.values():
            if o.kind != "monster":
                continue
            if o.state == "defeated":
                if self.tick >= o.data.get("respawn_tick", 0):
                    o.state = "roaming"
                    o.pos = tuple(o.data["home"])  # type: ignore[assignment]
                    self._bump(o)
                    evs.append(self._event("monster_respawned", target=o.id, pos=o.pos,
                                           importance=1, species=o.data["species"]))
                continue
            if o.state != "roaming" or self.tick % c.roam_every:
                continue
            dx, dy = self.rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)))
            nx, ny = o.pos[0] + dx, o.pos[1] + dy
            home = o.data["home"]
            if (zx <= nx < zx + zw and zy <= ny < zy + zh and abs(nx - home[0]) <= 4
                    and abs(ny - home[1]) <= 4 and self.grid.walkable((nx, ny))):
                if (nx, ny) != o.pos:
                    o.pos = (nx, ny)
                    self._bump(o)
        return evs

    def _daily_phase2(self: World) -> list[WorldEvent]:
        evs: list[WorldEvent] = []
        s = self.config.shops
        guild = next((g for g in self.objects.values() if g.kind == "guild_counter"), None)
        # village tax on hoarded savings: keeps the coin supply bounded and recirculates half
        for aid in sorted(self.actors):
            a = self.actors[aid]
            excess = a.inventory.get("coins", 0) - s.tax_threshold
            tax = math.floor(excess * s.tax_rate) if excess > 0 else 0
            if tax > 0:
                self._spend(a.inventory, "coins", tax)
                if guild is not None:
                    guild.data["treasury"] = guild.data.get("treasury", 0) + tax // 2
                self._bump(a)
                evs.append(self._event("tax_paid", actor=aid, importance=1, coins=tax))
        for o in self.objects.values():
            if o.kind == "guild_counter":
                master = self.actors.get(o.owner or "")
                wage = min(s.guild_wage, o.data.get("treasury", 0))
                if master is not None and wage > 0:
                    o.data["treasury"] -= wage
                    master.inventory["coins"] = master.inventory.get("coins", 0) + wage
                    self._bump(master)
                o.data["treasury"] = max(s.guild_treasury_floor,
                                         o.data.get("treasury", 0) + s.guild_stipend)
                self._bump(o)
        # expired villager requests / bounties give their escrow back
        for order in self.orders:
            if order.fulfilled_by is None and order.expires_day < self.day and order.escrow:
                poster = self.actors.get(order.poster_id or "")
                if order.kind == "bounty":
                    guild = next((g for g in self.objects.values()
                                  if g.kind == "guild_counter" and g.owner == order.poster_id),
                                 None)
                    if guild is not None:
                        guild.data["treasury"] = guild.data.get("treasury", 0) + order.escrow
                        self._bump(guild)
                elif poster is not None:
                    poster.inventory["coins"] = poster.inventory.get("coins", 0) + order.escrow
                    self._bump(poster)
                evs.append(self._event("request_expired", actor=order.poster_id, importance=1,
                                       order=order.id, refund=order.escrow))
                order.escrow = 0
        return evs

    # ------------------------------------------------------------------ engine hooks
    def swap_items(self: World, a_id: str, b_id: str, give_item: str, give_qty: int,
                   want_item: str, want_qty: int) -> tuple[bool, str, list[WorldEvent]]:
        """Atomic face-to-face barter agreed in conversation (``coins`` allowed)."""
        a, b = self.actors.get(a_id), self.actors.get(b_id)
        if a is None or b is None:
            return False, "no_actor", []
        if _cheb(a.pos, b.pos) > 2:
            return False, "too_far", []
        if give_qty < 1 or want_qty < 1:
            return False, "bad_qty", []
        if a.inventory.get(give_item, 0) < give_qty:
            return False, "offerer_lacks_item", []
        if b.inventory.get(want_item, 0) < want_qty:
            return False, "partner_lacks_item", []
        self._spend(a.inventory, give_item, give_qty)
        self._spend(b.inventory, want_item, want_qty)
        b.inventory[give_item] = b.inventory.get(give_item, 0) + give_qty
        a.inventory[want_item] = a.inventory.get(want_item, 0) + want_qty
        self._auto_equip(a)
        self._auto_equip(b)
        self._bump(a)
        self._bump(b)
        return True, "", [self._event("p2p_trade", actor=a.id, target=b.id, pos=a.pos,
                                      importance=3, give_item=give_item, give_qty=give_qty,
                                      want_item=want_item, want_qty=want_qty)]

    def coin_holdings(self: World) -> dict[str, int]:
        """Coins per villager plus the guild treasury (for supply / inequality metrics)."""
        out = {aid: a.inventory.get("coins", 0) for aid, a in self.actors.items()}
        for o in self.objects.values():
            if o.kind == "guild_counter":
                out[f"treasury:{o.id}"] = o.data.get("treasury", 0)
        return out

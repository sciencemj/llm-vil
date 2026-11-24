"""Phase 2 tests: profession/skill gating, crafting stations, P2P shops, escrowed requests,
conversation barter swaps, and tool durability & repair.
"""

from __future__ import annotations

from llm_vil.schemas.world import AtomicAction, WorldEvent
from llm_vil.world.world import World


def _awake(config) -> World:
    w = World(config)
    for aid in w.actors:
        w.apply(aid, AtomicAction(kind="wake"))
    return w


def _finish(w: World, aid: str, limit: int = 120) -> list[WorldEvent]:
    evs: list[WorldEvent] = []
    for _ in range(limit):
        if w.actors[aid].action is None:
            break
        evs.extend(w.tick_once())
    assert w.actors[aid].action is None
    return evs


def _place_adj(w: World, aid: str, oid: str) -> None:
    obj = w.objects[oid]
    spots = sorted(w.grid.neighbors8(obj.pos))
    assert spots, f"no walkable neighbor for {oid}"
    w.actors[aid].pos = spots[0]


def test_profession_skill_gating_and_all_stations(config):
    w = _awake(config)
    stations = {o.kind: o.id for o in w.objects.values()
                if o.kind in ("bakery_oven", "forge_anvil", "kiln", "cauldron")}
    assert set(stations) == {"bakery_oven", "forge_anvil", "kiln", "cauldron"}

    # Non-baker cannot bake bread at bakery_oven; baker (taeo) can.
    _place_adj(w, "mina", stations["bakery_oven"])
    w.actors["mina"].inventory.update({"wheat": 2, "wood": 1})
    bad = w.apply("mina", AtomicAction(kind="craft", target_id=stations["bakery_oven"],
                                       item="bread"))
    assert not bad.ok and bad.reason == "needs_skill"

    _place_adj(w, "taeo", stations["bakery_oven"])
    w.actors["taeo"].inventory.update({"wheat": 2, "wood": 1})
    assert w.apply("taeo", AtomicAction(kind="craft", target_id=stations["bakery_oven"],
                                        item="bread")).ok
    evs = _finish(w, "taeo")
    assert any(e.kind == "crafted" and e.data["output"] == "bread" for e in evs)
    assert w.actors["taeo"].inventory.get("bread", 0) >= 3

    # Woodcutter (junho) crafts charcoal at kiln.
    _place_adj(w, "junho", stations["kiln"])
    w.actors["junho"].inventory["wood"] = 3
    assert w.apply("junho", AtomicAction(kind="craft", target_id=stations["kiln"],
                                         item="charcoal")).ok
    _finish(w, "junho")
    assert w.actors["junho"].inventory.get("charcoal", 0) >= 2

    # Herbalist (haneul) crafts heal_potion at cauldron.
    _place_adj(w, "haneul", stations["cauldron"])
    w.actors["haneul"].inventory.update({"herb": 2, "slime_gel": 1, "wood": 1})
    assert w.apply("haneul", AtomicAction(kind="craft", target_id=stations["cauldron"],
                                          item="heal_potion")).ok
    _finish(w, "haneul")
    assert w.actors["haneul"].inventory.get("heal_potion", 0) >= 2

    # Blacksmith (gangcheol) crafts sword at forge_anvil; non-smith rejected.
    _place_adj(w, "bawi", stations["forge_anvil"])
    w.actors["bawi"].inventory.update({"iron_ore": 3, "coal": 1})
    assert w.apply("bawi", AtomicAction(kind="craft", target_id=stations["forge_anvil"],
                                        item="sword")).reason == "needs_skill"

    _place_adj(w, "gangcheol", stations["forge_anvil"])
    w.actors["gangcheol"].inventory.update({"iron_ore": 3, "coal": 1})
    assert w.apply("gangcheol", AtomicAction(kind="craft", target_id=stations["forge_anvil"],
                                             item="sword")).ok
    _finish(w, "gangcheol")
    assert (w.actors["gangcheol"].inventory.get("sword", 0) >= 1
            or w.actors["gangcheol"].equipment.get("weapon", {}).get("item") == "sword")


def test_shop_stock_buy_and_price_band(config):
    w = _awake(config)
    shop = next(o for o in w.objects.values()
                if o.kind == "shop_counter" and o.owner == "taeo")
    _place_adj(w, "taeo", shop.id)
    w.actors["taeo"].inventory["bread"] = 4

    # Non-owner cannot stock or set prices.
    _place_adj(w, "mina", shop.id)
    w.actors["mina"].inventory["bread"] = 2
    assert w.apply("mina", AtomicAction(kind="stock_shop", target_id=shop.id,
                                        item="bread", qty=1)).reason == "not_your_shop"
    assert w.apply("mina", AtomicAction(kind="set_prices", target_id=shop.id,
                                        item="bread", amount=8)).reason == "not_your_shop"

    # Owner stocks bread and sets price within 0.7-1.5x band.
    before_stock = shop.data["stock"].get("bread", 0)
    assert w.apply("taeo", AtomicAction(kind="stock_shop", target_id=shop.id,
                                        item="bread", qty=2)).ok
    _finish(w, "taeo")
    assert shop.data["stock"]["bread"] == before_stock + 2

    lo, hi = w.price_band("bread")
    assert w.apply("taeo", AtomicAction(kind="set_prices", target_id=shop.id,
                                        item="bread", amount=lo - 1)).reason.startswith(
        "price_out_of_band"
    )
    assert w.apply("taeo", AtomicAction(kind="set_prices", target_id=shop.id,
                                        item="bread", amount=hi + 1)).reason.startswith(
        "price_out_of_band"
    )
    assert w.apply("taeo", AtomicAction(kind="set_prices", target_id=shop.id,
                                        item="bread", amount=hi)).ok
    _finish(w, "taeo")
    assert shop.data["prices"]["bread"] == hi

    # Owner cannot buy from own shop; buyer (mina) buys atomically.
    assert w.apply("taeo", AtomicAction(kind="buy_from_shop", target_id=shop.id,
                                        item="bread", qty=1)).reason == "own_shop"
    taeo_coins = w.actors["taeo"].inventory.get("coins", 0)
    mina_coins = w.actors["mina"].inventory.get("coins", 0)
    assert mina_coins >= hi
    assert w.apply("mina", AtomicAction(kind="buy_from_shop", target_id=shop.id,
                                        item="bread", qty=1)).ok
    evs = _finish(w, "mina")
    assert any(e.kind == "shop_sale" and e.data["price"] == hi for e in evs)
    assert w.actors["mina"].inventory["coins"] == mina_coins - hi
    assert w.actors["taeo"].inventory["coins"] == taeo_coins + hi
    assert shop.data["stock"]["bread"] == before_stock + 1


def test_escrowed_request_fulfill_and_expiry_refund(config):
    w = _awake(config)
    board = next(o for o in w.objects.values() if o.kind == "notice_board")
    _place_adj(w, "taeo", board.id)
    coins0 = w.actors["taeo"].inventory["coins"]
    assert w.apply("taeo", AtomicAction(kind="post_request", target_id=board.id,
                                        item="wheat", qty=2, amount=10)).ok
    _finish(w, "taeo")
    assert w.actors["taeo"].inventory["coins"] == coins0 - 10
    req = next(o for o in w.orders if o.poster_id == "taeo" and o.item == "wheat")
    assert req.escrow == 10

    # Mina fulfills the request: wheat goes to taeo, escrow pays mina.
    _place_adj(w, "mina", board.id)
    w.actors["mina"].inventory["wheat"] = 3
    mina_coins0 = w.actors["mina"].inventory["coins"]
    taeo_wheat0 = w.actors["taeo"].inventory.get("wheat", 0)
    assert w.apply("mina", AtomicAction(kind="fulfill_order", target_id=board.id,
                                        item=req.id)).ok
    _finish(w, "mina")
    assert w.actors["taeo"].inventory["wheat"] == taeo_wheat0 + 2
    assert w.actors["mina"].inventory["coins"] == mina_coins0 + 10
    assert req.escrow == 0 and req.fulfilled_by == "mina"

    # Unfulfilled request refunds escrow upon expiry.
    coins1 = w.actors["taeo"].inventory["coins"]
    assert w.apply("taeo", AtomicAction(kind="post_request", target_id=board.id,
                                        item="wood", qty=1, amount=6)).ok
    _finish(w, "taeo")
    exp_req = next(o for o in w.orders if o.poster_id == "taeo" and o.item == "wood")
    assert w.actors["taeo"].inventory["coins"] == coins1 - 6
    exp_req.expires_day = w.day - 1
    evs = w._daily_phase2()
    assert any(e.kind == "request_expired" and e.data["refund"] == 6 for e in evs)
    assert exp_req.escrow == 0
    assert w.actors["taeo"].inventory["coins"] >= coins1 - 6  # refunded (minus any wealth tax)


def test_atomic_p2p_barter_swap(config):
    w = _awake(config)
    w.actors["mina"].pos = (15, 15)
    w.actors["junho"].pos = (16, 15)
    w.actors["mina"].inventory["turnip"] = 3
    w.actors["junho"].inventory["wood"] = 4
    ok, reason, evs = w.swap_items("mina", "junho", "turnip", 2, "wood", 3)
    assert ok and reason == ""
    assert any(e.kind == "p2p_trade" for e in evs)
    assert w.actors["mina"].inventory["turnip"] == 1
    assert w.actors["mina"].inventory["wood"] >= 3
    assert w.actors["junho"].inventory["turnip"] >= 2
    assert w.actors["junho"].inventory["wood"] == 1

    # Fails cleanly when too far or lacking item.
    w.actors["junho"].pos = (25, 15)
    ok2, reason2, _ = w.swap_items("mina", "junho", "turnip", 1, "wood", 1)
    assert not ok2 and reason2 == "too_far"


def test_tool_durability_soft_gate_and_repair(config):
    w = _awake(config)
    forge = next(o for o in w.objects.values() if o.kind == "forge_anvil")
    ore = next(o for o in w.objects.values() if o.kind == "ore_node")

    # Wear bawi's pickaxe to 0; hand-mining still succeeds (soft gate, no deadlock).
    pick = w.actors["bawi"].equipment["tool"]
    assert pick["item"] == "pickaxe"
    pick["durability"] = 0
    _place_adj(w, "bawi", ore.id)
    res_no_tool = w.apply("bawi", AtomicAction(kind="mine", target_id=ore.id))
    assert res_no_tool.ok
    _finish(w, "bawi")

    # Repair at forge_anvil costs repair_fee for non-smith and restores max_durability.
    _place_adj(w, "bawi", forge.id)
    coins0 = w.actors["bawi"].inventory["coins"]
    fee = config.shops.repair_fee
    assert coins0 >= fee
    assert w.apply("bawi", AtomicAction(kind="repair_tool", target_id=forge.id,
                                        item="tool")).ok
    evs = _finish(w, "bawi")
    assert any(e.kind == "tool_repaired" and e.data["fee"] == fee for e in evs)
    assert w.actors["bawi"].equipment["tool"]["durability"] == pick["max_durability"]
    assert w.actors["bawi"].inventory["coins"] == coins0 - fee


def test_stock_shop_rejects_coins_and_multi_qty_sale_records_total(config):
    w = _awake(config)
    shop = next(o for o in w.objects.values()
                if o.kind == "shop_counter" and o.owner == "taeo")
    _place_adj(w, "taeo", shop.id)
    bad = w.apply("taeo", AtomicAction(kind="stock_shop", target_id=shop.id,
                                       item="coins", qty=1))
    assert not bad.ok and bad.reason == "bad_item"

    shop.data["stock"]["bread"] = 5
    price = w.shop_price(shop, "bread")
    _place_adj(w, "mina", shop.id)
    w.actors["mina"].inventory["coins"] = price * 3
    assert w.apply("mina", AtomicAction(kind="buy_from_shop", target_id=shop.id,
                                        item="bread", qty=2)).ok
    evs = _finish(w, "mina")
    sale = next(e for e in evs if e.kind == "shop_sale")
    assert sale.data["price"] == price and sale.data["total"] == price * 2


def test_bounties_do_not_block_guildmaster_requests_and_refund_bumps_poster(config):
    w = _awake(config)
    guild = next(o for o in w.objects.values() if o.kind == "guild_counter")
    board = next(o for o in w.objects.values() if o.kind == "notice_board")
    _place_adj(w, "hyeonwoo", guild.id)
    for species in ("slime", "boar"):
        assert w.apply("hyeonwoo", AtomicAction(kind="post_bounty", target_id=guild.id,
                                                item=species, amount=10)).ok
        _finish(w, "hyeonwoo")

    # 2 open bounties must not count against MAX_OPEN_REQUESTS_PER_VILLAGER.
    _place_adj(w, "hyeonwoo", board.id)
    w.actors["hyeonwoo"].inventory["coins"] = 50
    assert w.apply("hyeonwoo", AtomicAction(kind="post_request", target_id=board.id,
                                            item="bread", qty=1, amount=10)).ok
    _finish(w, "hyeonwoo")
    req = next(o for o in w.orders if o.kind == "order" and o.poster_id == "hyeonwoo")
    v_before = w.actors["hyeonwoo"].version
    req.expires_day = w.day - 1
    w._daily_phase2()
    assert w.actors["hyeonwoo"].version > v_before


from __future__ import annotations

from llm_vil.llm.protocol import dumps
from llm_vil.schemas.world import AtomicAction, Weather
from llm_vil.world.world import World


def awake_world(config) -> World:
    w = World(config)
    for aid in w.actors:
        assert w.apply(aid, AtomicAction(kind="wake")).ok
    return w


def run_until_idle(w: World, aid: str, limit: int = 200) -> list:
    events = []
    for _ in range(limit):
        events.extend(w.tick_once())
        if w.actors[aid].action is None:
            return events
    raise AssertionError("action never finished")


def test_sleeping_actor_rejected_and_unknown(config):
    w = World(config)
    assert w.apply("mina", AtomicAction(kind="eat", item="berry")).reason == "asleep"
    assert w.apply("ghost", AtomicAction(kind="wait")).reason == "unknown_actor"
    assert w.apply("mina", AtomicAction(kind="talk", target_id="junho")).reason == "engine_only"


def test_validity_gate_rejects_bad_intents(config):
    w = awake_world(config)
    assert w.apply("mina", AtomicAction(kind="chop", target_id="tree_01")).reason == "too_far"
    assert w.apply("mina", AtomicAction(kind="chop", target_id="bed_mina")).reason == (
        "not_afforded")
    assert w.apply("mina", AtomicAction(kind="eat", item="wood")).reason == "not_food"
    assert w.apply("mina", AtomicAction(kind="sleep", target_id="bed_junho")).reason in (
        "too_far", "not_your_bed")
    assert w.apply("mina", AtomicAction(kind="wait", ticks=5)).ok
    assert w.apply("mina", AtomicAction(kind="wait")).reason == "busy"


def test_move_then_chop_yields_wood_and_stump(config):
    w = awake_world(config)
    assert w.apply("junho", AtomicAction(kind="move_to", target_id="tree_03")).ok
    assert w.actors["junho"].activity == "walking"
    run_until_idle(w, "junho")
    tree = w.objects["tree_03"]
    assert max(abs(tree.pos[0] - w.actors["junho"].pos[0]),
               abs(tree.pos[1] - w.actors["junho"].pos[1])) <= 1
    wood = w.actors["junho"].inventory.get("wood", 0)
    v0 = tree.version
    assert w.apply("junho", AtomicAction(kind="chop", target_id="tree_03")).ok
    evs = run_until_idle(w, "junho")
    assert any(e.kind == "chopped" for e in evs)
    assert w.objects["tree_03"].state == "stump"
    assert w.objects["tree_03"].version > v0
    assert w.actors["junho"].inventory["wood"] >= wood + 3


def test_forest_reserve_blocks_extinction(config):
    w = awake_world(config)
    trees = [o for o in w.objects.values() if o.kind == "tree"]
    for t in trees[config.economy.forest_reserve:]:
        t.state = "stump"
    target = trees[0]
    w.actors["junho"].pos = (target.pos[0], target.pos[1] + 1)
    assert w.apply("junho", AtomicAction(kind="chop", target_id=target.id)).reason == (
        "forest_reserve")


def test_trees_and_bushes_regrow(config):
    w = awake_world(config)
    w.objects["tree_01"].state = "stump"
    w.objects["tree_01"].data["stage_days"] = 0
    w.objects["bush_nw"].state = "empty"
    w.objects["bush_nw"].data["stage_days"] = 0
    for _ in range(4 * config.ticks_per_day):
        w.tick_once()
    assert w.objects["tree_01"].state == "mature"
    assert w.objects["bush_nw"].state == "ripe"


def test_farming_cycle_and_rain_waters(config):
    w = awake_world(config)
    plot = w.objects["plot_01"]
    w.actors["mina"].pos = (plot.pos[0], plot.pos[1] + 1)
    for act in (AtomicAction(kind="till", target_id="plot_01"),
                AtomicAction(kind="plant", target_id="plot_01", item="turnip_seed")):
        assert w.apply("mina", act).ok, act
        run_until_idle(w, "mina")
    assert plot.state in ("planted", "watered")
    w.set_weather(Weather.rain)
    assert plot.state == "watered"
    for _ in range(2 * config.ticks_per_day):
        if plot.state == "planted":
            plot.state = "watered"
        w.tick_once()
    assert plot.state == "ready"
    seeds = w.actors["mina"].inventory.get("turnip_seed", 0)
    assert w.apply("mina", AtomicAction(kind="harvest", target_id="plot_01")).ok
    evs = run_until_idle(w, "mina")
    harvest = next(e for e in evs if e.kind == "harvested")
    assert harvest.data["items"]["turnip"] == 3
    assert w.actors["mina"].inventory["turnip_seed"] >= seeds + 1  # never bricks farming
    assert plot.state == "tilled"


def test_take_from_others_chest_is_witnessed_and_discovered(config):
    w = awake_world(config)
    chest = w.objects["chest_haneul"]
    w.actors["doyun"].pos = (chest.pos[0], chest.pos[1] + 1)
    w.actors["mina"].pos = (chest.pos[0] - 2, chest.pos[1] + 1)  # bystander
    assert w.apply("doyun", AtomicAction(kind="take", target_id="chest_haneul", item="berry")).ok
    evs = run_until_idle(w, "doyun")
    taken = next(e for e in evs if e.kind == "chest_taken")
    assert taken.actor_id == "doyun" and taken.target_id == "haneul"
    assert "mina" in taken.witnesses and "doyun" not in taken.witnesses
    w.actors["haneul"].pos = (chest.pos[0] + 1, chest.pos[1] + 1)
    assert w.apply("haneul", AtomicAction(kind="take", target_id="chest_haneul",
                                          item="berry")).ok
    evs = run_until_idle(w, "haneul")
    found = next(e for e in evs if e.kind == "chest_missing_discovered")
    assert found.target_id == "haneul"
    assert found.data["missing"] == [{"item": "berry", "qty": 1}]


def test_hunger_is_non_lethal_and_slows_work(config):
    w = awake_world(config)
    w.actors["mina"].inventory["berry"] = 2
    w.actors["mina"].hunger = 99.99
    for _ in range(500):
        w.tick_once()
    assert w.actors["mina"].hunger == 100.0
    assert w.apply("mina", AtomicAction(kind="eat", item="berry")).end_tick == w.tick + 10
    assert w.actors["mina"].mood < 65


def test_merchant_and_orders(config):
    w = awake_world(config)
    assert len(w.orders) == config.economy.orders_per_day
    while w.minute != config.economy.merchant_open_minute:
        w.tick_once()
    assert w.merchant_present  # day 0 is a merchant day
    stall = w.objects["stall_main"]
    w.actors["doyun"].pos = (stall.pos[0], stall.pos[1] + 1)
    coins = w.actors["doyun"].inventory["coins"]
    assert w.apply("doyun", AtomicAction(kind="trade", target_id="stall_main", item="turnip_seed",
                                         op="buy", qty=2)).ok
    run_until_idle(w, "doyun")
    assert w.actors["doyun"].inventory["coins"] == coins - 4


def test_snapshot_restore_roundtrip(config):
    w = awake_world(config)
    w.apply("junho", AtomicAction(kind="move_to", target_id="tree_05"))
    for _ in range(700):
        w.tick_once()
    snap = w.snapshot()
    w2 = World.restore(config, snap)
    assert dumps(w2.snapshot()) == dumps(snap)
    for _ in range(1500):
        w.tick_once()
        w2.tick_once()
    assert dumps(w2.snapshot()) == dumps(w.snapshot())

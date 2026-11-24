"""Phase 2 tests: Mosswood monster roam & respawn, seeded combat, non-lethal fainted recovery,
and Adventurers' Guild bounties + drop buy-back/resale.
"""

from __future__ import annotations

from llm_vil.schemas.world import AtomicAction, WorldEvent
from llm_vil.world.world import World


def _awake(config, seed: int = 42) -> World:
    w = World(config, seed=seed)
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


def test_monsters_roam_inside_mosswood_and_respawn(config):
    w = _awake(config)
    zx, zy, zw, zh = config.combat.zone
    monsters = [o for o in w.objects.values() if o.kind == "monster"]
    assert len(monsters) == 6
    initial_positions = {m.id: m.pos for m in monsters}
    moved = False
    for _ in range(config.combat.roam_every * 20):
        w.tick_once()
        for m in monsters:
            assert zx <= m.pos[0] < zx + zw and zy <= m.pos[1] < zy + zh
            if m.pos != initial_positions[m.id]:
                moved = True
    assert moved

    # Defeated monster respawns at its home tile after respawn_minutes.
    slime = next(m for m in monsters if m.data["species"] == "slime")
    slime.state = "defeated"
    slime.data["respawn_tick"] = w.tick + 5
    respawned = False
    for _ in range(10):
        evs = w.tick_once()
        if any(e.kind == "monster_respawned" and e.target_id == slime.id for e in evs):
            respawned = True
            break
    assert respawned and slime.state == "roaming" and list(slime.pos) == slime.data["home"]


def test_seeded_fight_win_drops_and_durability(config):
    # Find a seed where rina wins against slime_1, then verify determinism.
    w = _awake(config, seed=7)
    slime = w.objects["slime_1"]
    _place_adj(w, "rina", slime.id)
    dur0 = w.actors["rina"].equipment["weapon"]["durability"]
    assert w.apply("rina", AtomicAction(kind="fight", target_id=slime.id)).ok
    evs = _finish(w, "rina")
    # Repeat with identical seed and confirm identical outcome.
    w2 = _awake(config, seed=7)
    _place_adj(w2, "rina", "slime_1")
    assert w2.apply("rina", AtomicAction(kind="fight", target_id="slime_1")).ok
    evs2 = _finish(w2, "rina")
    assert [e.kind for e in evs] == [e.kind for e in evs2]

    # Force a guaranteed win via high rng roll check (re-try seeds until win).
    won = False
    for seed in range(1, 20):
        wt = _awake(config, seed=seed)
        _place_adj(wt, "rina", "slime_1")
        dur_before = wt.actors["rina"].equipment["weapon"]["durability"]
        assert wt.apply("rina", AtomicAction(kind="fight", target_id="slime_1")).ok
        fevs = _finish(wt, "rina")
        if any(e.kind == "monster_defeated" for e in fevs):
            assert wt.objects["slime_1"].state == "defeated"
            assert wt.actors["rina"].inventory.get("slime_gel", 0) >= 1
            assert wt.actors["rina"].equipment["weapon"]["durability"] == dur_before - 1
            won = True
            break
    assert won and dur0 > 0


def test_non_lethal_faint_recovers_at_clinic(config):
    w = _awake(config)
    rina = w.actors["rina"]
    rina.inventory["slime_gel"] = 4
    rina.inventory["coins"] = 50
    rina.hp = 10.0  # below faint_hp
    evs = w._faint(rina)
    faint_ev = next(e for e in evs if e.kind == "fainted")
    assert faint_ev.importance == 7
    # Lost half of unbanked monster drops, kept coins, recovered HP at clinic.
    assert rina.inventory["slime_gel"] == 2
    assert rina.inventory["coins"] == 50
    assert rina.hp == config.combat.recover_hp
    clinic = w.objects[config.combat.clinic]
    assert max(abs(rina.pos[0] - clinic.pos[0]), abs(rina.pos[1] - clinic.pos[1])) <= 1


def test_guild_bounty_drop_buyback_and_resale(config):
    w = _awake(config)
    guild = next(o for o in w.objects.values() if o.kind == "guild_counter")
    treasury0 = guild.data["treasury"]

    # Guildmaster posts a slime bounty escrowed from guild treasury.
    _place_adj(w, "hyeonwoo", guild.id)
    assert w.apply("hyeonwoo", AtomicAction(kind="post_bounty", target_id=guild.id,
                                            item="slime", amount=15)).ok
    _finish(w, "hyeonwoo")
    assert guild.data["treasury"] == treasury0 - 15
    bounty = next(o for o in w.orders if o.kind == "bounty" and o.item == "slime")
    assert bounty.escrow == 15

    # Completing bounty on slime defeat pays adventurer.
    rina_coins0 = w.actors["rina"].inventory["coins"]
    bevs = w._complete_bounty(w.actors["rina"], "slime", w.objects["slime_1"])
    assert any(e.kind == "bounty_completed" and e.data["reward"] == 15 for e in bevs)
    assert w.actors["rina"].inventory["coins"] == rina_coins0 + 15
    assert bounty.fulfilled_by == "rina" and bounty.escrow == 0

    # Adventurer sells monster drops to guild counter; guild stocks them for resale.
    _place_adj(w, "rina", guild.id)
    w.actors["rina"].inventory["slime_gel"] = 3
    t_before = guild.data["treasury"]
    c_before = w.actors["rina"].inventory["coins"]
    assert w.apply("rina", AtomicAction(kind="sell_drops", target_id=guild.id,
                                        item="slime_gel", qty=2)).ok
    sevs = _finish(w, "rina")
    paid = w._drop_price("slime_gel") * 2
    assert any(e.kind == "drops_sold" and e.data["coins"] == paid for e in sevs)
    assert guild.data["treasury"] == t_before - paid
    assert w.actors["rina"].inventory["coins"] == c_before + paid
    assert guild.data["stock"].get("slime_gel", 0) >= 2

    # Herbalist buys slime_gel from guild counter; coins return to guild treasury.
    _place_adj(w, "haneul", guild.id)
    price = w.shop_price(guild, "slime_gel")
    t_resale = guild.data["treasury"]
    assert w.apply("haneul", AtomicAction(kind="buy_from_shop", target_id=guild.id,
                                          item="slime_gel", qty=1)).ok
    _finish(w, "haneul")
    assert guild.data["treasury"] == t_resale + price
    assert w.actors["haneul"].inventory.get("slime_gel", 0) >= 1


def test_broken_armor_does_not_reduce_damage(config):
    slime_dmg = config.monsters["slime"].damage
    won = False
    for seed in range(1, 20):
        w = _awake(config, seed=seed)
        rina = w.actors["rina"]
        rina.hp = 100.0
        rina.equipment["armor"] = {"item": "armor", "durability": 0, "max_durability": 30}
        _place_adj(w, "rina", "slime_1")
        assert w.apply("rina", AtomicAction(kind="fight", target_id="slime_1")).ok
        evs = _finish(w, "rina")
        if any(e.kind == "monster_defeated" for e in evs):
            # Broken armor must NOT halve damage or decrement durability below 0.
            assert rina.equipment["armor"]["durability"] == 0
            # Account for small awake hp_regen during the fight duration.
            assert rina.hp <= 100.0 - slime_dmg + 2.0
            assert rina.hp < 100.0 - slime_dmg * 0.5
            won = True
            break
    assert won


async def test_fainted_clears_routine_and_hp_retreat_boundary(config):
    from llm_vil.agent.routines import HP_RETREAT, RoutineContext, _hunt
    from llm_vil.engine import SimEngine
    from llm_vil.llm import DemoBackend
    from llm_vil.schemas.agent import RoutineChoice

    e = SimEngine(config, DemoBackend())
    w = e.world
    w.apply("rina", AtomicAction(kind="wake"))
    rina = w.actors["rina"]
    agent = e.agents["rina"]
    agent.routine = RoutineChoice(routine="hunt", target="mosswood", item="slime", qty=3)
    agent.progress = {"fights": 1}

    # Trigger a fight loss that drops HP below faint_hp on completion.
    slime = w.objects["slime_1"]
    _place_adj(w, "rina", slime.id)
    rina.hp = config.combat.faint_hp + 1.0
    assert w.apply("rina", AtomicAction(kind="fight", target_id=slime.id)).ok
    assert rina.action is not None
    # Force guaranteed loss by lowering rng or setting hp so even win faints.
    rina.hp = config.combat.faint_hp + 0.5
    rina.equipment.pop("armor", None)
    # Step until fight completes and rina faints.
    end = rina.action.end_tick
    while w.tick < end:
        await e.step()
    assert agent.routine is None or agent.routine.routine != "hunt"
    assert agent.progress == {} or "fights" not in agent.progress

    # _hunt retreats at exact HP_RETREAT boundary (<=, not <).
    rina.hp = HP_RETREAT
    view = w.perceive("rina")
    ctx = RoutineContext(view=view, known=agent.known, config=config, spec=agent.spec)
    choice = RoutineChoice(routine="hunt", target="mosswood", item="any", qty=2)
    assert _hunt(choice, {}, ctx) is None


def test_fight_failure_in_complete_releases_engaged_monster(config):
    w = _awake(config)
    slime = w.objects["slime_1"]
    _place_adj(w, "rina", slime.id)
    assert w.apply("rina", AtomicAction(kind="fight", target_id=slime.id)).ok
    assert slime.state == "engaged" and slime.data.get("engaged_by") == "rina"

    # Make rina too hurt mid-fight so _complete fails the validity re-check.
    w.actors["rina"].hp = 1.0
    evs = _finish(w, "rina")
    assert any(e.kind == "action_failed" and e.data.get("reason") == "too_hurt" for e in evs)
    assert slime.state == "roaming"
    assert "engaged_by" not in slime.data


def test_sell_drops_respects_guild_stock_cap_on_qty(config):
    w = _awake(config)
    guild = next(o for o in w.objects.values() if o.kind == "guild_counter")
    cap = config.shops.guild_stock_cap
    guild.data["stock"]["slime_gel"] = cap - 1
    _place_adj(w, "rina", guild.id)
    w.actors["rina"].inventory["slime_gel"] = 3
    bad = w.apply("rina", AtomicAction(kind="sell_drops", target_id=guild.id,
                                       item="slime_gel", qty=2))
    assert not bad.ok and bad.reason == "guild_full"
    assert w.apply("rina", AtomicAction(kind="sell_drops", target_id=guild.id,
                                        item="slime_gel", qty=1)).ok
    _finish(w, "rina")
    assert guild.data["stock"]["slime_gel"] == cap


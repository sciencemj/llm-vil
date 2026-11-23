"""30-sim-day headless DemoBackend soak test (M1 acceptance)."""

import json
import time

import pytest

from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import dumps

DAYS = 30


@pytest.mark.soak
async def test_thirty_day_soak(config, tmp_path):
    store = SaveStore(tmp_path / "soak.sqlite")
    e = SimEngine(config, DemoBackend(), store=store, checkpoint_every_days=7)
    t0 = time.perf_counter()
    min_stats: dict[str, int] = {}
    for _ in range(DAYS):
        await e.run_days(1)
        for k, v in e.world.stats().items():
            min_stats[k] = min(min_stats.get(k, v), v)
    elapsed = time.perf_counter() - t0
    e.flush()
    s = e.summary()
    m = s["metrics"]
    rel_deltas = {
        f"{a.id}->{t}": {ax: getattr(r, ax) for ax in ("familiarity", "affection", "trust",
                                                        "respect")}
        for a in e.agents.values() for t, r in a.relationships.rels.items()
    }
    report = {"elapsed_s": round(elapsed, 2), "days": DAYS, "metrics": m, "world": s["world"],
              "min_world": min_stats, "router": s["router"],
              "relationship_magnitude": s["relationship_magnitude"],
              "checkpoints": store.count("checkpoints"), "events": store.count("events"),
              "archived": store.count("memory_archive"), "tape": store.count("tape"),
              "goals": {a.id: a.goals for a in e.agents.values()}}
    print("\nSOAK_REPORT " + dumps(report))
    (tmp_path / "soak_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    rel_path = tmp_path / "relationships.json"
    rel_path.write_text(json.dumps(rel_deltas, ensure_ascii=False, indent=2), encoding="utf-8")

    assert e.world.day >= DAYS
    # 0 unhandled exceptions
    assert s["errors"] == [] and m["errors"] == 0
    # 0 villagers stuck > 2 sim-hours
    assert m["stuck_incidents"] == 0 and m["max_stuck_minutes"] <= 120
    # no starvation spiral (never pinned at hunger >= 95 for a whole sim-day)
    assert m["max_starving_minutes"] < 1440
    # 0 permanent resource extinctions
    assert min_stats["trees_mature"] >= config.economy.forest_reserve
    assert s["world"]["trees_total"] >= 12 and min_stats["bushes_total"] == 4
    assert min_stats["seeds_total"] + min_stats["plots_growing"] > 0
    assert s["world"]["plots_growing"] > 0
    # active economy & social life
    assert m["wood_chopped"] > 0 and m["crops_harvested"] > 0
    assert m["meals_cooked"] > 0 and m["meals_eaten"] > 0
    assert m["conversations_completed"] > 0 and m["conversation_turns"] > 0
    assert m["reflections"] >= DAYS * len(config.villagers) * 0.8
    assert m["weekly_syntheses"] >= len(config.villagers) * 3
    assert m["archive_queries"] > 0 and m["archive_hits"] > 0 and m["archived_memories"] > 0
    assert s["relationship_magnitude"] > 1.0
    # M4: gossip spreads organically (no God rumors in this run) and coins circulate
    assert m["hearsay_shared"] > 0
    assert s["world"]["coins_total"] < 500, s["world"]["coins_total"]
    assert m["trades"] > 30
    assert all(a.goals for a in e.agents.values())
    assert store.count("checkpoints") >= 4
    store.close()

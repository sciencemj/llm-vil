"""Bit-identical snapshot/restore: checkpoint at day 2, restore, re-run, compare."""

from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import dumps

DAY = 1440


async def test_checkpoint_restore_is_bit_identical(config, tmp_path):
    path = tmp_path / "village.sqlite"
    end_tick = 4 * DAY + 600  # past the first memory archival (hot window = 3 days)

    store = SaveStore(path)
    a = SimEngine(config, DemoBackend(), store=store)
    await a.run_until(2 * DAY)
    a.checkpoint()
    await a.run_until(end_tick)
    a.flush()
    snap_a = dumps(a.snapshot())
    counts_a = {t: store.count(t) for t in ("events", "memory_archive", "tape")}
    assert counts_a["memory_archive"] > 0
    store.close()

    store_b = SaveStore(path)
    b = SimEngine.from_checkpoint(config, DemoBackend(), store_b, at_or_before=2 * DAY)
    assert b.tick == 2 * DAY
    assert store_b.count("events") < counts_a["events"]  # log truncated to the checkpoint
    await b.run_until(end_tick)
    b.flush()
    snap_b = dumps(b.snapshot())
    assert snap_b == snap_a
    assert {t: store_b.count(t) for t in ("events", "memory_archive", "tape")} == counts_a
    store_b.close()


async def test_two_fresh_runs_are_identical(config):
    e1, e2 = SimEngine(config, DemoBackend()), SimEngine(config, DemoBackend())
    await e1.run_days(1)
    await e2.run_days(1)
    assert dumps(e1.snapshot()) == dumps(e2.snapshot())

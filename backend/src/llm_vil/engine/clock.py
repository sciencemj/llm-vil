"""Cognition-aware clock.

At ``speed`` x, 1 real second = ``sim_minutes_per_real_second * speed`` sim minutes. While
the oldest in-flight cognition has waited < ``t_max_seconds`` the sim keeps advancing;
past that, advancement dilates (blocks) until a cognition resolves. ``t_max_seconds <= 0``
means "always wait" — the deterministic headless mode used by tests and replays.
When every villager sleeps, pacing is skipped (sleep fast-forward).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class CognitionClock:
    def __init__(self, *, speed: float | None = None, sim_minutes_per_real_second: float = 2.0,
                 t_max_seconds: float = 0.0,
                 time_fn: Callable[[], float] = time.monotonic) -> None:
        self.speed = speed  # None = unpaced (headless, as fast as possible)
        self.sim_minutes_per_real_second = sim_minutes_per_real_second
        self.t_max_seconds = t_max_seconds
        self.time_fn = time_fn
        self.dilated_ticks = 0
        self.fast_forward_ticks = 0
        self.max_cognition_wait = 0.0
        self._next_due: float | None = None

    @property
    def deterministic(self) -> bool:
        return self.t_max_seconds <= 0

    def set_speed(self, speed: float | None) -> None:
        self.speed = speed
        self._next_due = None

    def must_wait(self, inflight_started: list[float]) -> bool:
        if not inflight_started:
            return False
        if self.deterministic:
            return True
        waited = self.time_fn() - min(inflight_started)
        self.max_cognition_wait = max(self.max_cognition_wait, waited)
        return waited >= self.t_max_seconds

    async def pace(self, all_sleeping: bool) -> None:
        if all_sleeping:
            self.fast_forward_ticks += 1
            self._next_due = None
            return
        if self.speed is None or self.speed <= 0:
            return
        interval = 1.0 / (self.sim_minutes_per_real_second * self.speed)
        now = self.time_fn()
        self._next_due = (self._next_due or now) + interval
        delay = self._next_due - now
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -1.0:  # fell far behind (e.g. dilation): resync instead of bursting
            self._next_due = now

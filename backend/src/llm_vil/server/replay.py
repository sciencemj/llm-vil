"""Replay recording: one compact visual frame per sim tick, viewable without touching the
live simulation.

Frames hold only what the renderer needs (clock, villager pose/activity/speech/needs,
object states, the tick's timeline events). They are buffered in memory and written in
batches through the single-writer ``SaveStore``; memory-only runs keep a bounded deque.
"""

from __future__ import annotations

import bisect
import json
from collections import deque
from typing import Any

from ..engine import SaveStore

DEFAULT_CAP = 10_000      # ticks kept (~7 sim days)
FLUSH_EVERY = 30          # frames per batched write
SPEECH_TICKS = 4          # a line of dialogue stays visible this many ticks
EVENT_WINDOW = 240        # ticks of timeline gathered for a replayed frame


class ReplayRecorder:
    def __init__(self, store: SaveStore | None, cap: int = DEFAULT_CAP) -> None:
        self.store = store
        self.cap = cap
        self._pending: list[tuple[int, int, int, str]] = []
        self._memory: deque[tuple[int, int, int, str]] = deque(maxlen=cap)  # memory-only

    # ------------------------------------------------------------ write
    def record(self, tick: int, day: int, minute: int, frame: dict[str, Any]) -> None:
        row = (tick, day, minute, json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
        if self.store is None:
            if self._memory and self._memory[-1][0] >= tick:  # rewound: drop the future
                self.truncate_after(tick - 1)
            self._memory.append(row)
            return
        self._pending.append(row)
        if len(self._pending) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self.store is not None and self._pending:
            self.store.write_frames(self._pending, self.cap)
            self._pending = []

    def truncate_after(self, tick: int) -> None:
        """Forget frames of an abandoned future (store rows are truncated by the store)."""
        self._pending = [f for f in self._pending if f[0] <= tick]
        while self._memory and self._memory[-1][0] > tick:
            self._memory.pop()

    def reset(self) -> None:
        self._pending = []
        self._memory.clear()

    # ------------------------------------------------------------ read
    def ticks(self) -> list[int]:
        if self.store is None:
            return [f[0] for f in self._memory]
        stored = self.store.frame_ticks()
        pend = [f[0] for f in self._pending]
        return sorted(set(stored) | set(pend))

    def frame_at(self, tick: int) -> tuple[int, dict[str, Any]] | None:
        rows = self._rows_upto(tick, 1)
        if not rows:
            return None
        t, raw = rows[-1]
        return t, json.loads(raw)

    def events_upto(self, tick: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _, raw in self._between(tick - EVENT_WINDOW, tick):
            if '"ev":[]' in raw:  # most ticks have no events: skip the JSON parse
                continue
            out.extend(json.loads(raw).get("ev", []))
        return out[-120:]

    def _between(self, lo: int, hi: int) -> list[tuple[int, str]]:
        if self.store is None:
            return [(f[0], f[3]) for f in self._memory if lo <= f[0] <= hi]
        rows = dict(self.store.frames_between(lo, hi))
        rows.update({f[0]: f[3] for f in self._pending if lo <= f[0] <= hi})
        return sorted(rows.items())

    def _rows_upto(self, tick: int, n: int) -> list[tuple[int, str]]:
        if self.store is None:
            ticks = [f[0] for f in self._memory]
            i = bisect.bisect_right(ticks, tick)
            return [(self._memory[j][0], self._memory[j][3]) for j in range(max(0, i - n), i)]
        best: tuple[int, str] | None = None
        hit = self.store.frame_at(tick)
        if hit is not None:
            best = hit
        for f in self._pending:
            if f[0] <= tick and (best is None or f[0] > best[0]):
                best = (f[0], f[3])
        return [best] if best else []

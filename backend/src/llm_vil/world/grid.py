from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Iterable

Tile = tuple[int, int]
_DIRS: tuple[Tile, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
_NEIGHBORS8: tuple[Tile, ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)


def facing_toward(src: Tile, dst: Tile, default: int = 90) -> int:
    """Degrees snapped to 45°; 0 = east (+x), 90 = south (+y)."""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if dx == 0 and dy == 0:
        return default
    return (round(math.degrees(math.atan2(dy, dx)) / 45) * 45) % 360


class TileGrid:
    """Static walkability grid with deterministic 4-connected A*."""

    def __init__(self, width: int, height: int, blocked: Iterable[Tile] = ()) -> None:
        self.width = width
        self.height = height
        self.blocked: set[Tile] = set(blocked)

    def in_bounds(self, t: Tile) -> bool:
        return 0 <= t[0] < self.width and 0 <= t[1] < self.height

    def walkable(self, t: Tile) -> bool:
        return self.in_bounds(t) and t not in self.blocked

    def neighbors8(self, t: Tile) -> list[Tile]:
        cands = [(t[0] + dx, t[1] + dy) for dx, dy in _NEIGHBORS8]
        return [c for c in cands if self.walkable(c)]

    def astar(self, start: Tile, goals: Iterable[Tile]) -> list[Tile] | None:
        """Shortest path from ``start`` to any goal (excluding start). Ties broken by tile."""
        goal_set = {g for g in goals if self.walkable(g)}
        if not goal_set:
            return None
        if start in goal_set:
            return []
        ordered_goals = sorted(goal_set)

        def h(t: Tile) -> int:
            return min(abs(t[0] - g[0]) + abs(t[1] - g[1]) for g in ordered_goals)

        open_heap: list[tuple[int, int, Tile]] = [(h(start), 0, start)]
        came: dict[Tile, Tile] = {}
        best: dict[Tile, int] = {start: 0}
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in goal_set:
                path = [cur]
                while path[-1] in came and came[path[-1]] != start:
                    path.append(came[path[-1]])
                path.reverse()
                return path
            if g > best.get(cur, 1 << 30):
                continue
            for dx, dy in _DIRS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not self.walkable(nxt):
                    continue
                ng = g + 1
                if ng < best.get(nxt, 1 << 30):
                    best[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
        return None

    def nearest_walkable(self, t: Tile) -> Tile | None:
        if self.walkable(t):
            return t
        seen = {t}
        q: deque[Tile] = deque([t])
        while q:
            cur = q.popleft()
            for dx, dy in _DIRS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in seen or not self.in_bounds(nxt):
                    continue
                if self.walkable(nxt):
                    return nxt
                seen.add(nxt)
                q.append(nxt)
        return None

    @staticmethod
    def line_of_sight(a: Tile, b: Tile, opaque: set[Tile]) -> bool:
        """Bresenham; endpoints never block."""
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while (x0, y0) != (x1, y1):
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
            if (x0, y0) != (x1, y1) and (x0, y0) in opaque:
                return False
        return True

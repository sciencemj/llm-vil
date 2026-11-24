"""Change-driven delta messages (ready for the M2 WebSocket stream)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .base import Frozen
from .world import Facing, TilePos


class PathMoveDelta(Frozen):
    type: Literal["path_move"] = "path_move"
    actor_id: str
    start_tick: int
    step_ticks: int
    waypoints: list[TilePos]
    facing: Facing


class ObjectChangeDelta(Frozen):
    type: Literal["object_change"] = "object_change"
    object_id: str
    state: str
    version: int
    data: dict[str, Any] = Field(default_factory=dict)
    pos: TilePos | None = None  # moving objects (monsters)


class ConversationDelta(Frozen):
    type: Literal["conversation"] = "conversation"
    session_id: str
    participants: list[str]
    state: str
    speaker_id: str | None = None
    utterance: str | None = None
    emotion: str | None = None


class ClockDelta(Frozen):
    type: Literal["clock"] = "clock"
    tick: int
    day: int
    minute: int
    speed: float | None
    dilated: bool = False
    fast_forward: bool = False


class GodEventDelta(Frozen):
    type: Literal["god_event"] = "god_event"
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


Delta = PathMoveDelta | ObjectChangeDelta | ConversationDelta | ClockDelta | GodEventDelta

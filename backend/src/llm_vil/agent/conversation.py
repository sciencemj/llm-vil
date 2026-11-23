"""``ConversationSession`` state machine.

invite -> accept | decline -> active_turns (max N, 1 LLM call per turn)
       -> end {silence, max_turns, interrupt, departure}

Sessions only hold facts; they return a ``ConversationOutcome`` that each participant's
``Agent`` applies to its own memory and relationships.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..schemas.agent import ConversationEnd, ConversationOutcome, ConversationTurn
from ..schemas.base import Mutable

SessionState = Literal["invited", "active", "ended"]


class ConversationError(RuntimeError):
    pass


class ConversationSession(Mutable):
    id: str
    initiator_id: str
    target_id: str
    topic: str
    start_tick: int
    max_turns: int = 6
    state: SessionState = "invited"
    turns: list[ConversationTurn] = Field(default_factory=list)
    end_reason: ConversationEnd | None = None
    end_tick: int | None = None
    next_turn_tick: int = 0

    @property
    def participants(self) -> list[str]:
        return [self.initiator_id, self.target_id]

    @property
    def next_speaker(self) -> str:
        return self.initiator_id if len(self.turns) % 2 == 0 else self.target_id

    def partner_of(self, actor_id: str) -> str:
        return self.target_id if actor_id == self.initiator_id else self.initiator_id

    def accept(self, tick: int) -> None:
        if self.state != "invited":
            raise ConversationError(f"cannot accept in state {self.state}")
        self.state = "active"
        self.next_turn_tick = tick

    def decline(self, tick: int) -> None:
        if self.state != "invited":
            raise ConversationError(f"cannot decline in state {self.state}")
        self._end("declined", tick)

    def add_turn(self, turn: ConversationTurn, ticks_per_turn: int) -> None:
        if self.state != "active":
            raise ConversationError(f"cannot add turn in state {self.state}")
        if turn.speaker_id != self.next_speaker:
            raise ConversationError(f"not {turn.speaker_id}'s turn")
        self.turns.append(turn)
        self.next_turn_tick = turn.tick + ticks_per_turn
        if turn.end_conversation and len(self.turns) >= 2:
            self._end("silence", turn.tick)
        elif len(self.turns) >= self.max_turns:
            self._end("max_turns", turn.tick)

    def interrupt(self, tick: int, reason: ConversationEnd = "interrupt") -> None:
        if self.state != "ended":
            self._end(reason, tick)

    def _end(self, reason: ConversationEnd, tick: int) -> None:
        self.state = "ended"
        self.end_reason = reason
        self.end_tick = tick

    def outcome(self) -> ConversationOutcome:
        if self.state != "ended" or self.end_reason is None:
            raise ConversationError("session not ended")
        mean: dict[str, float] = {}
        for pid in self.participants:
            vals = [t.valence for t in self.turns if t.speaker_id == pid]
            mean[pid] = round(sum(vals) / len(vals), 4) if vals else 0.0
        gifts = [(t.speaker_id, t.listener_id, t.gift_item) for t in self.turns
                 if t.gift_item != "none"]
        return ConversationOutcome(
            session_id=self.id, participants=self.participants, initiator_id=self.initiator_id,
            topic=self.topic, turns=list(self.turns), end_reason=self.end_reason,
            start_tick=self.start_tick, end_tick=self.end_tick or self.start_tick,
            mean_valence=mean, gifts=gifts,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

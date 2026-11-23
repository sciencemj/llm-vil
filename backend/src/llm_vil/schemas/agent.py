from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Frozen

RoutineName = Literal[
    "tend_farm",
    "chop_wood",
    "forage",
    "cook_meal",
    "eat_food",
    "store_or_take",
    "visit_and_talk",
    "post_or_fulfill_order",
    "trade",
    "sleep",
    "rest",
    "wander",
]

FacingCue = Literal["none", "face_partner", "look_away", "survey_area"]
Expression = Literal["neutral", "happy", "sad", "angry", "surprised", "tired", "thoughtful"]
Emotion = Literal["neutral", "joy", "warmth", "worry", "annoyance", "surprise", "sadness"]

MemoryKind = Literal[
    "observation",
    "conversation",
    "reflection",
    "hearsay",
    "plan",
    "summary",
    "fact",
    "goal",
    "external_event",
]


class AgentSpec(Frozen):
    """Immutable persona core. Never overwritten by reflection or synthesis."""

    id: str
    name: str
    age: int
    occupation: str
    traits: list[str]
    persona: str
    wake_minute: int
    sleep_minute: int
    home_area: str
    bed_id: str
    chest_id: str
    plots: list[str]
    start_pos: tuple[int, int]
    inventory: dict[str, int] = Field(default_factory=dict)
    work_bias: dict[str, float] = Field(default_factory=dict)
    generosity: float = 0.5
    impulsiveness: float = 0.1
    sociability: float = 0.5


class PlanBlock(Frozen):
    start_minute: int
    end_minute: int
    routine: RoutineName
    note: str = ""


class RoutineChoice(Frozen):
    routine: RoutineName
    target: str = "none"
    item: str = "none"
    qty: int = 1
    op: str = "none"
    topic: str = ""
    reason: str = ""
    facing_cue: FacingCue = "none"
    expression: Expression = "neutral"


class MemoryRecord(Frozen):
    id: str
    owner_id: str
    tick: int
    kind: MemoryKind
    text: str
    importance: int = Field(ge=1, le=10)
    valence: float = Field(default=0.0, ge=-1, le=1)
    subjects: list[str] = Field(default_factory=list)
    source_actor_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    prompt_version: str = ""
    embed_model: str = ""
    embedding: list[float] = Field(default_factory=list)


class Relationship(Frozen):
    """Directed edge owner -> target. ``a -> b`` is independent from ``b -> a``."""

    target_id: str
    familiarity: float = Field(default=0.0, ge=0, le=1)
    affection: float = Field(default=0.0, ge=-1, le=1)
    trust: float = Field(default=0.0, ge=-1, le=1)
    respect: float = Field(default=0.0, ge=-1, le=1)
    summary: str = ""
    recent_key_events: list[str] = Field(default_factory=list)


class ConversationTurn(Frozen):
    speaker_id: str
    listener_id: str
    tick: int
    utterance: str
    end_conversation: bool = False
    emotion: Emotion = "neutral"
    valence: float = Field(default=0.0, ge=-1, le=1)
    importance: int = Field(default=3, ge=1, le=10)
    facing_cue: FacingCue = "face_partner"
    gift_item: str = "none"
    share_memory_id: str = "none"


ConversationEnd = Literal["silence", "max_turns", "interrupt", "departure", "declined"]


class ConversationOutcome(Frozen):
    session_id: str
    participants: list[str]
    initiator_id: str
    topic: str
    turns: list[ConversationTurn]
    end_reason: ConversationEnd
    start_tick: int
    end_tick: int
    mean_valence: dict[str, float] = Field(default_factory=dict)
    gifts: list[tuple[str, str, str]] = Field(default_factory=list)  # (giver, receiver, item)

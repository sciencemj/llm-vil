"""Flat LLM reply schemas (small enums, no unions / $ref) shared by every provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas.agent import Emotion, Expression, FacingCue, RoutineName


class _Reply(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RoutineReply(_Reply):
    routine: RoutineName
    target: str = "none"
    item: str = "none"
    qty: int = Field(default=1, ge=1, le=20)
    op: Literal["none", "store", "take", "buy", "sell"] = "none"
    topic: str = ""
    reason: str = ""
    facing_cue: FacingCue = "none"
    expression: Expression = "neutral"


class PlanBlockReply(_Reply):
    start_minute: int = Field(ge=0, lt=1440)
    end_minute: int = Field(ge=0, le=1440)
    routine: RoutineName
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> PlanBlockReply:
        if self.end_minute <= self.start_minute:
            raise ValueError("end_minute must be greater than start_minute")
        return self


class PlanReply(_Reply):
    blocks: list[PlanBlockReply] = Field(min_length=1, max_length=12)


class TurnReply(_Reply):
    utterance: str = Field(min_length=1, max_length=240)
    end_conversation: bool = False
    emotion: Emotion = "neutral"
    valence: float = Field(default=0.0, ge=-1, le=1)
    importance: int = Field(default=3, ge=1, le=10)
    facing_cue: FacingCue = "face_partner"
    gift_item: str = "none"
    share_memory_id: str = "none"


class ReflectionItem(_Reply):
    text: str = Field(min_length=1)
    importance: int = Field(default=6, ge=1, le=10)
    evidence_ids: list[str] = Field(default_factory=list)


class ReflectReply(_Reply):
    reflections: list[ReflectionItem] = Field(min_length=1, max_length=3)
    facts: list[str] = Field(default_factory=list, max_length=3)


class GoalItem(_Reply):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class GoalsReply(_Reply):
    goals: list[GoalItem] = Field(min_length=1, max_length=3)


def flat_schema(model: type[BaseModel], enums: dict[str, list[str]] | None = None) -> dict:
    """JSON schema with every ``$ref`` inlined; optional per-field enum narrowing."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def inline(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    schema = inline(raw)
    assert isinstance(schema, dict)
    for field, values in (enums or {}).items():
        if field in schema.get("properties", {}):
            schema["properties"][field]["enum"] = values
    return schema

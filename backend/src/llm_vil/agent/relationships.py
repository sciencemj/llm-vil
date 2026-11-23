"""Directed 4-axis relationships with bounded, deterministic update math."""

from __future__ import annotations

from typing import Any

from ..schemas.agent import Relationship

AXES = ("familiarity", "affection", "trust", "respect")
BOUNDS = {"familiarity": (0.0, 1.0), "affection": (-1.0, 1.0), "trust": (-1.0, 1.0),
          "respect": (-1.0, 1.0)}

EVENT_DELTAS: dict[str, dict[str, float]] = {
    "gift_received": {"affection": 0.12, "familiarity": 0.03},
    "gift_given": {"affection": 0.04, "familiarity": 0.02},
    "fair_trade": {"trust": 0.10, "familiarity": 0.02},
    "witnessed_theft": {"trust": -0.35, "affection": -0.10, "respect": -0.10},
    "pleasant_chat": {"familiarity": 0.05, "affection": 0.03},
    "neutral_chat": {"familiarity": 0.05},
    "unpleasant_chat": {"familiarity": 0.05, "affection": -0.05},
    "declined_invite": {"affection": -0.02},
    "hearsay_theft": {"trust": -0.12},
    "saw_working": {"respect": 0.01, "familiarity": 0.005},
}
APPRAISAL_CLAMP = 0.15
SUMMARY_REFRESH_DELTA = 0.20
DAILY_DECAY = 0.98
FAMILIARITY_DECAY = 0.998
MAX_KEY_EVENTS = 5


def _clamp(axis: str, v: float) -> float:
    lo, hi = BOUNDS[axis]
    return round(max(lo, min(hi, v)), 6)


def _soft(current: float, delta: float) -> float:
    """Diminishing returns: a push in the direction the axis already leans shrinks as the axis
    nears its bound, so repeated routine events cannot saturate a relationship."""
    if current * delta > 0:
        return delta * (1.0 - abs(current))
    return delta


def describe(r: Relationship, name: str) -> str:
    def word(v: float, pos: str, neg: str) -> str:
        return pos if v >= 0.25 else neg if v <= -0.25 else ""

    parts = [w for w in (
        "잘 아는 사이" if r.familiarity >= 0.4 else "아직 낯선 사이" if r.familiarity < 0.1 else "",
        word(r.affection, "호감이 감", "마음이 불편함"),
        word(r.trust, "믿을 만함", "믿기 어려움"),
        word(r.respect, "존경스러움", "얕보게 됨"),
    ) if w]
    return f"{name}: " + (", ".join(parts) if parts else "그럭저럭 무난한 사이")


class RelationshipStore:
    def __init__(self, owner_id: str, others: dict[str, str]) -> None:
        self.owner_id = owner_id
        self.names = dict(others)  # id -> display name
        self.rels: dict[str, Relationship] = {
            oid: Relationship(target_id=oid, summary=describe(Relationship(target_id=oid), name))
            for oid, name in others.items()
        }
        self.drift: dict[str, float] = {oid: 0.0 for oid in others}

    def get(self, target: str) -> Relationship | None:
        return self.rels.get(target)

    def _update(self, target: str, deltas: dict[str, float], note: str | None) -> None:
        r = self.rels.get(target)
        if r is None:
            return
        upd: dict[str, Any] = {a: _clamp(a, getattr(r, a) + _soft(getattr(r, a), d))
                               for a, d in deltas.items()}
        moved = sum(abs(upd[a] - getattr(r, a)) for a in upd)
        events = list(r.recent_key_events)
        if note:
            events = (events + [note])[-MAX_KEY_EVENTS:]
        upd["recent_key_events"] = events
        r = r.model_copy(update=upd)
        self.drift[target] = self.drift.get(target, 0.0) + moved
        if self.drift[target] >= SUMMARY_REFRESH_DELTA:
            r = r.model_copy(update={"summary": describe(r, self.names[target])})
            self.drift[target] = 0.0
        self.rels[target] = r

    def apply_event(self, target: str, event: str, *, scale: float = 1.0,
                    note: str | None = None) -> None:
        deltas = {a: d * scale for a, d in EVENT_DELTAS[event].items()}
        self._update(target, deltas, note)

    def appraise(self, target: str, deltas: dict[str, float], note: str | None = None) -> None:
        """LLM-sourced appraisal, clamped to +-0.15 per axis to prevent drift/saturation."""
        clamped = {
            a: max(-APPRAISAL_CLAMP, min(APPRAISAL_CLAMP, float(d)))
            for a, d in deltas.items() if a in AXES
        }
        self._update(target, clamped, note)

    def daily_decay(self) -> None:
        for oid, r in self.rels.items():
            self.rels[oid] = r.model_copy(update={
                "familiarity": _clamp("familiarity", r.familiarity * FAMILIARITY_DECAY),
                "affection": _clamp("affection", r.affection * DAILY_DECAY),
                "trust": _clamp("trust", r.trust * DAILY_DECAY),
                "respect": _clamp("respect", r.respect * DAILY_DECAY),
            })

    def refresh_summaries(self) -> None:
        for oid, r in self.rels.items():
            self.rels[oid] = r.model_copy(update={"summary": describe(r, self.names[oid])})
            self.drift[oid] = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "rels": {k: v.model_dump(mode="json") for k, v in self.rels.items()},
            "drift": dict(self.drift),
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.rels = {k: Relationship.model_validate(v) for k, v in data["rels"].items()}
        self.drift = dict(data["drift"])

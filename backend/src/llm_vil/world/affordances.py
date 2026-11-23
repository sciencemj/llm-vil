from __future__ import annotations

from ..schemas.config import AffordanceDef, ObjectKindDef, VillageConfig


class AffordanceRegistry:
    """Data-driven object affordances loaded from ``village.yaml``."""

    def __init__(self, config: VillageConfig) -> None:
        self._kinds: dict[str, ObjectKindDef] = dict(config.kinds)

    def kind(self, kind: str) -> ObjectKindDef:
        return self._kinds[kind]

    def get(self, kind: str, affordance: str) -> AffordanceDef | None:
        k = self._kinds.get(kind)
        return k.affordances.get(affordance) if k else None

    def affordances_for(self, kind: str, state: str | None = None) -> list[str]:
        k = self._kinds.get(kind)
        if k is None:
            return []
        return [
            name
            for name, a in k.affordances.items()
            if state is None or not a.from_states or state in a.from_states
        ]

    def state_allows(self, kind: str, affordance: str, state: str) -> bool:
        a = self.get(kind, affordance)
        return a is not None and (not a.from_states or state in a.from_states)

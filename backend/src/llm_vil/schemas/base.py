from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable I/O model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Mutable(BaseModel):
    """Mutable internal state model (snapshot-able via ``model_dump(mode="json")``)."""

    model_config = ConfigDict(extra="forbid")

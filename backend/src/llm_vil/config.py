"""YAML config loading. Depends only on `schemas`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas.config import VillageConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_village(path: str | Path | None = None) -> VillageConfig:
    p = Path(path) if path else CONFIG_DIR / "village.yaml"
    with p.open(encoding="utf-8") as fh:
        return VillageConfig.model_validate(yaml.safe_load(fh))


def load_llm_profiles(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_DIR / "llm.yaml"
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)

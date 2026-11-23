from __future__ import annotations

import pytest

from llm_vil.config import load_village
from llm_vil.schemas.config import VillageConfig


@pytest.fixture(scope="session")
def config() -> VillageConfig:
    return load_village()

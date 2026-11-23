"""Provider-agnostic LLM runtime: protocol, router (repair loop, rate gate, budget, tape),
local embeddings and the deterministic `DemoBackend`."""

from .demo import DemoBackend
from .protocol import (
    HashingEmbedder,
    LanguageModel,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    SharedRateGate,
    TokenBudget,
    dumps,
    extract_json,
    strip_think,
)

__all__ = [
    "DemoBackend",
    "HashingEmbedder",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "LanguageModel",
    "SharedRateGate",
    "TokenBudget",
    "dumps",
    "extract_json",
    "strip_think",
]

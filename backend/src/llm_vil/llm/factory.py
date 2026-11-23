"""Build ``(LanguageModel, Embedder)`` from ``config/llm.yaml`` + profile names + env/.env.

Secrets: only ``api_key_env`` *names* are in YAML; values come from the environment (or a
``.env`` file loaded without overriding existing variables). ``public_info`` never contains
a key, so it is safe to serve over REST / WebSocket.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import CONFIG_DIR
from ._http import resolve_api_key
from .anthropic_adapter import AnthropicAdapter
from .config_store import ConfigPaths, load_merged_llm_config
from .demo import DemoBackend
from .embeddings import (
    CachingEmbedder,
    DemoCachedEmbedder,
    LocalFastEmbedder,
    OpenAICompatEmbedder,
    VectorCache,
)
from .gemini_adapter import GeminiAdapter
from .openai_compat import OpenAICompatAdapter
from .protocol import LanguageModel, Pricing, TokenBudget

DEFAULT_ENV_FILE = CONFIG_DIR.parent / ".env"


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Minimal ``.env`` loader (KEY=VALUE, # comments). Never overrides the environment.
    Returns the names it set (never values)."""
    p = Path(path) if path else DEFAULT_ENV_FILE
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = _dotenv_value(value)
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _dotenv_value(raw: str) -> str:
    """``"a # b"`` keeps its hash; unquoted values drop an inline `` # comment``; only a
    matching pair of outer quotes is removed."""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] == '"':
        # inverse of config_store._env_quote: \\ -> \ and \" -> "
        return re.sub(r'\\(["\\])', r"\1", v[1:-1])
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1]  # single quotes are literal
    if " #" in v:
        v = v.split(" #", 1)[0]
    return v.strip()


@dataclass
class LLMBundle:
    llm: LanguageModel
    embedder: CachingEmbedder
    profile: str
    embed_profile: str
    pricing: dict[str, Pricing] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None  # profile timeout; runtime.timeout_seconds is the fallback

    @property
    def llm_timeout(self) -> float:
        return float(self.timeout_s or self.runtime.get("timeout_seconds") or 30)

    @property
    def repair_retries(self) -> int:
        value = self.runtime.get("json_repair_retries")
        return 2 if value is None else int(value)

    def token_budget(self) -> TokenBudget:
        def opt(key: str) -> int | None:
            v = self.runtime.get(key)
            return None if v is None else int(v)

        return TokenBudget(per_day=opt("token_budget_per_day"),
                           per_run=opt("token_budget_per_run"))

    async def aclose(self) -> None:
        """Release HTTP clients held by the model and the embedder."""
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()
        self.embedder.close()

    def public_info(self) -> dict[str, Any]:
        describe = getattr(self.llm, "describe", None)
        info = describe() if callable(describe) else {
            "provider": "demo", "fast_model": getattr(self.llm, "model", "?"),
            "smart_model": getattr(self.llm, "model", "?"),
        }
        return {"profile": self.profile, "embed_profile": self.embed_profile, **info}


def _pricing(raw: dict[str, Any] | None) -> dict[str, Pricing]:
    return {k: Pricing(**v) for k, v in (raw or {}).items()}


_FROM_ENV: Any = object()


def build_llm(name: str, prof: dict[str, Any],
              transport: httpx.AsyncBaseTransport | None = None,
              api_key: Any = _FROM_ENV) -> LanguageModel:
    """``api_key`` overrides the env lookup (write-only probe of an unsaved key)."""
    provider = prof.get("provider")
    if provider == "demo":
        return DemoBackend(simulated_latency_ms=float(prof.get("simulated_latency_ms") or 0))
    common: dict[str, Any] = {
        "base_url": prof["base_url"],
        "fast_model": prof.get("fast_model") or prof.get("model"),
        "smart_model": prof.get("smart_model"),
        "api_key": api_key if api_key not in (_FROM_ENV, None) else resolve_api_key(
            prof.get("api_key_env"),
            required=provider in ("anthropic", "gemini") or bool(prof.get("api_key_required"))),
        "extra_headers": prof.get("extra_headers") or {},
        "timeout_s": float(prof.get("timeout_s") or 60),
        "transport": transport,
    }
    if prof.get("max_tokens"):
        common["max_tokens"] = int(prof["max_tokens"])
    if provider == "openai_compat":
        return OpenAICompatAdapter(structured=prof.get("structured", "json_schema"), **common)
    if provider == "anthropic":
        return AnthropicAdapter(effort_fast=prof.get("effort_fast", "low"),
                                effort_smart=prof.get("effort_smart", "medium"),
                                refusal_fallbacks=bool(prof.get("refusal_fallbacks", True)),
                                **common)
    if provider == "gemini":
        return GeminiAdapter(**common)
    raise ValueError(f"profile {name!r}: unknown provider {provider!r}")


def build_embedder(name: str, prof: dict[str, Any], cache: VectorCache | None = None,
                   transport: httpx.BaseTransport | None = None) -> CachingEmbedder:
    provider = prof.get("provider")
    if provider == "demo":
        return DemoCachedEmbedder(dim=int(prof.get("dim") or 64), cache=cache)
    if provider == "local":
        return LocalFastEmbedder(model=prof["model"], dim=int(prof["dim"]), cache=cache,
                                 cache_dir=prof.get("cache_dir"))
    if provider == "openai_compat":
        return OpenAICompatEmbedder(
            base_url=prof["base_url"], model=prof["model"], dim=int(prof["dim"]),
            api_key=resolve_api_key(prof.get("api_key_env"), required=False),
            extra_headers=prof.get("extra_headers"), cache=cache, transport=transport,
        )
    raise ValueError(f"embeddings profile {name!r}: unknown provider {provider!r}")


def build_bundle(llm_profile: str | None = None, embed_profile: str | None = None, *,
                 config_path: str | Path | None = None, env_file: str | Path | None = None,
                 cache: VectorCache | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 llm: LanguageModel | None = None) -> LLMBundle:
    """``llm`` overrides the profile's model (tests); no adapter is built in that case."""
    paths = ConfigPaths.from_args(config_path, env_file)
    load_dotenv(paths.env_file)
    cfg = load_merged_llm_config(paths)  # llm.yaml + gitignored llm.local.yaml
    profiles = cfg["profiles"]
    name = llm_profile or os.environ.get("LLM_VIL_PROFILE") or cfg.get("default_profile", "demo")
    if name not in profiles:
        raise ValueError(f"unknown LLM profile {name!r}; choose from {sorted(profiles)}")
    prof = profiles[name]
    emb_cfg = cfg.get("embeddings", {})
    emb_name = (embed_profile or os.environ.get("LLM_VIL_EMBED_PROFILE")
                or prof.get("embeddings") or emb_cfg.get("default", "demo"))
    emb_profiles = emb_cfg.get("profiles", {})
    if emb_name not in emb_profiles:
        raise ValueError(f"unknown embeddings profile {emb_name!r}; "
                         f"choose from {sorted(emb_profiles)}")
    return LLMBundle(
        llm=llm if llm is not None else build_llm(name, prof, transport),
        embedder=build_embedder(emb_name, emb_profiles[emb_name], cache),
        profile=name, embed_profile=emb_name, pricing=_pricing(prof.get("pricing")),
        runtime=dict(cfg.get("runtime") or {}),
        timeout_s=float(prof["timeout_s"]) if prof.get("timeout_s") else None,
    )

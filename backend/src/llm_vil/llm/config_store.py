"""Provider configuration store.

* ``config/llm.yaml``        - shipped defaults (committed)
* ``config/llm.local.yaml``  - non-secret local overrides (gitignored), merged on top
* ``backend/.env``           - API keys only (gitignored, chmod 600, atomic writes)

Nothing in this module ever returns a plaintext key: summaries expose ``has_api_key`` and a
masked hint (``****1234``), and every error string is scrubbed of known secret values.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import CONFIG_DIR
from ._http import is_llm_key_env

PROVIDERS = ("demo", "openai_compat", "anthropic", "gemini")
STRUCTURED = ("json_schema", "json_object", "tool_call", "prompt_only", "native")
EDITABLE_FIELDS = ("provider", "base_url", "fast_model", "smart_model", "structured",
                   "timeout_s", "api_key_env", "embeddings")
# provider -> structured modes its adapter actually implements
STRUCTURED_BY_PROVIDER = {
    "demo": ("native",),
    "openai_compat": ("json_schema", "json_object", "prompt_only"),
    "anthropic": ("tool_call",),
    "gemini": ("json_schema",),
}
ENV_HINT = "api_key_env must be an LLM key variable like MYPROVIDER_API_KEY"
PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"]}


class ConfigError(ValueError):
    """User-facing validation error (safe to show: never contains secrets)."""


@dataclass(frozen=True)
class ConfigPaths:
    base_yaml: Path = CONFIG_DIR / "llm.yaml"
    env_file: Path = CONFIG_DIR.parent / ".env"

    @property
    def local_yaml(self) -> Path:
        return self.base_yaml.with_name("llm.local.yaml")

    @classmethod
    def from_args(cls, base_yaml: str | Path | None = None,
                  env_file: str | Path | None = None) -> ConfigPaths:
        d = cls()
        return cls(Path(base_yaml) if base_yaml else d.base_yaml,
                   Path(env_file) if env_file else d.env_file)


# ---------------------------------------------------------------- yaml merge / save
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML; fix or delete it") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping")
    return data


def merge_configs(base: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    out.setdefault("profiles", {})
    for name, fields in (local.get("profiles") or {}).items():
        out["profiles"][name] = {**out["profiles"].get(name, {}), **(fields or {})}
    if local.get("default_profile"):
        out["default_profile"] = local["default_profile"]
    emb_local = local.get("embeddings") or {}
    if emb_local.get("default"):
        out.setdefault("embeddings", {})["default"] = emb_local["default"]
    return out


def load_merged_llm_config(paths: ConfigPaths | None = None) -> dict[str, Any]:
    p = paths or ConfigPaths()
    return merge_configs(_read_yaml(p.base_yaml), _read_yaml(p.local_yaml))


def _atomic_write(path: Path, text: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def validate_fields(fields: dict[str, Any], *, embed_profiles: list[str],
                    existing: dict[str, Any] | None) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in EDITABLE_FIELDS:
            raise ConfigError(f"field {key!r} cannot be edited")
        if value is None or value == "":
            if key == "api_key_env":
                clean[key] = None  # explicit "no key" (e.g. local Ollama)
            continue
        if key == "provider":
            if value not in PROVIDERS:
                raise ConfigError(f"provider must be one of {list(PROVIDERS)}")
        elif key == "structured":
            if value not in STRUCTURED:
                raise ConfigError(f"structured must be one of {list(STRUCTURED)}")
        elif key == "base_url":
            value = str(value).strip()
            if not re.match(r"^https?://[^\s/]+", value):
                raise ConfigError("base_url must start with http:// or https://")
            if "@" in value.split("//", 1)[1].split("/", 1)[0]:
                raise ConfigError("put credentials in the API key, not in base_url")
        elif key == "timeout_s":
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError("timeout_s must be a number") from exc
            if not 1 <= value <= 600:
                raise ConfigError("timeout_s must be between 1 and 600")
        elif key == "api_key_env":
            if not is_llm_key_env(str(value)):
                raise ConfigError(ENV_HINT)
        elif key == "embeddings":
            if value not in embed_profiles:
                raise ConfigError(f"embeddings must be one of {embed_profiles}")
        elif key in ("fast_model", "smart_model"):
            value = str(value).strip()
            if not value or len(value) > 200 or any(c.isspace() for c in value):
                raise ConfigError(f"{key} must be a model id without spaces")
        clean[key] = value
    if existing is None and "provider" not in clean:
        raise ConfigError("a new profile needs a provider")
    provider = clean.get("provider") or (existing or {}).get("provider")
    mode = clean.get("structured")
    if mode and provider in STRUCTURED_BY_PROVIDER and mode not in STRUCTURED_BY_PROVIDER[provider]:
        raise ConfigError(f"{provider} supports structured modes "
                          f"{list(STRUCTURED_BY_PROVIDER[provider])}")
    if provider in ("openai_compat", "anthropic", "gemini"):
        merged = {**(existing or {}), **clean}
        if not merged.get("base_url"):
            raise ConfigError(f"{provider} needs a base_url")
        if not (merged.get("fast_model") or merged.get("model")):
            raise ConfigError(f"{provider} needs a fast_model")
    return clean


def save_profile_override(profile: str, fields: dict[str, Any], *, set_default: bool = False,
                          paths: ConfigPaths | None = None) -> dict[str, Any]:
    """Validate and write non-secret profile fields to ``llm.local.yaml``."""
    p = paths or ConfigPaths()
    if not PROFILE_NAME.match(profile):
        raise ConfigError("profile name: lowercase letters, digits and _ only")
    merged = load_merged_llm_config(p)
    embed_profiles = sorted((merged.get("embeddings") or {}).get("profiles", {}))
    existing = merged.get("profiles", {}).get(profile)
    clean = validate_fields(fields, embed_profiles=embed_profiles, existing=existing)
    local = _read_yaml(p.local_yaml)
    local.setdefault("profiles", {})
    local["profiles"][profile] = {**(local["profiles"].get(profile) or {}), **clean}
    if set_default:
        local["default_profile"] = profile
    header = ("# Local, non-secret overrides merged over llm.yaml (gitignored).\n"
              "# API keys live in backend/.env only.\n")
    _atomic_write(p.local_yaml, header + yaml.safe_dump(local, allow_unicode=True,
                                                        sort_keys=True), 0o644)
    return load_merged_llm_config(p)["profiles"][profile]


# ---------------------------------------------------------------- .env keys
def _env_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_\-.:/+=]*", value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_env_key(env_name: str, value: str, paths: ConfigPaths | None = None) -> None:
    """Atomically set ``env_name`` in ``.env`` (mode 0600), keeping every other line."""
    p = paths or ConfigPaths()
    validate_key_env(env_name)
    value = validate_api_key(value)
    lines = p.env_file.read_text(encoding="utf-8").splitlines() if p.env_file.is_file() else []
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(env_name)}\s*=")
    new_line = f"{env_name}={_env_quote(value)}"
    replaced = False
    out = []
    for line in lines:
        if pattern.match(line):
            if not replaced:
                out.append(new_line)
                replaced = True
            continue  # drop duplicates of the same key
        out.append(line)
    if not replaced:
        out.append(new_line)
    _atomic_write(p.env_file, "\n".join(out) + "\n", 0o600)
    os.environ[env_name] = value


def validate_key_env(env_name: str | None) -> str:
    if not env_name or not is_llm_key_env(env_name):
        raise ConfigError(ENV_HINT)
    return env_name


def validate_api_key(value: str) -> str:
    value = (value or "").strip()
    if not value or len(value) > 4096 or any(c in value for c in "\r\n\0"):
        raise ConfigError("API key must be a single non-empty line")
    return value


def save_profile_and_key(profile: str, fields: dict[str, Any], api_key: str | None, *,
                         set_default: bool = False,
                         paths: ConfigPaths | None = None) -> dict[str, Any]:
    """Validate *everything* first, then write: an invalid key or env name never leaves
    ``llm.local.yaml`` half-updated. The key is written before the profile so a profile
    never points at a key write that failed."""
    p = paths or ConfigPaths()
    if not PROFILE_NAME.match(profile):
        raise ConfigError("profile name: lowercase letters, digits and _ only")
    merged = load_merged_llm_config(p)
    embed_profiles = sorted((merged.get("embeddings") or {}).get("profiles", {}))
    existing = merged.get("profiles", {}).get(profile)
    clean = validate_fields(fields, embed_profiles=embed_profiles, existing=existing)
    env = clean["api_key_env"] if "api_key_env" in clean else (existing or {}).get("api_key_env")
    if api_key:
        if not env:
            raise ConfigError("set api_key_env before saving an API key")
        validate_key_env(env)
        key = validate_api_key(api_key)
        save_env_key(env, key, p)
    return save_profile_override(profile, fields, set_default=set_default, paths=p)


def mask_key(value: str | None) -> str | None:
    if not value:
        return None
    return f"****{value[-4:]}" if len(value) >= 8 else "****"


def known_secrets(config: dict[str, Any]) -> list[str]:
    names = {prof.get("api_key_env") for prof in (config.get("profiles") or {}).values()}
    names |= {e.get("api_key_env")
              for e in ((config.get("embeddings") or {}).get("profiles") or {}).values()}
    return [os.environ[n] for n in sorted(x for x in names if is_llm_key_env(x))
            if os.environ.get(n)]


def scrub(text: str, secrets: list[str]) -> str:
    for s in sorted({x for x in secrets if x and len(x) >= 4}, key=len, reverse=True):
        text = text.replace(s, "***")
    return re.sub(r"(?i)(bearer\s+|x-api-key[:=]\s*|x-goog-api-key[:=]\s*)\S+", r"\1***", text)


# ---------------------------------------------------------------- summaries
def profile_summary(name: str, prof: dict[str, Any]) -> dict[str, Any]:
    from ._http import public_url

    env = prof.get("api_key_env")
    key = os.environ.get(env) if is_llm_key_env(env) else None  # never peek at other vars
    return {
        "name": name,
        "provider": prof.get("provider"),
        "base_url": public_url(prof["base_url"]) if prof.get("base_url") else None,
        "fast_model": prof.get("fast_model") or prof.get("model"),
        "smart_model": prof.get("smart_model") or prof.get("fast_model") or prof.get("model"),
        "structured": prof.get("structured"),
        "timeout_s": prof.get("timeout_s"),
        "api_key_env": env,
        "needs_key": prof.get("provider") in ("anthropic", "gemini") or (
            prof.get("provider") == "openai_compat" and bool(env)),
        "has_api_key": bool(key),
        "key_hint": mask_key(key),
        "embeddings": prof.get("embeddings"),
    }


def config_summary(active_profile: str | None, active_embed_profile: str | None,
                   paths: ConfigPaths | None = None) -> dict[str, Any]:
    from .factory import load_dotenv

    p = paths or ConfigPaths()
    load_dotenv(p.env_file)
    cfg = load_merged_llm_config(p)
    emb = cfg.get("embeddings") or {}
    return {
        "active_profile": active_profile,
        "active_embed_profile": active_embed_profile,
        "default_profile": cfg.get("default_profile", "demo"),
        "default_embed_profile": emb.get("default"),
        "profiles": [profile_summary(n, prof)
                     for n, prof in sorted((cfg.get("profiles") or {}).items())],
        "embed_profiles": [{"name": n, "provider": e.get("provider"), "model": e.get("model"),
                            "dim": e.get("dim")}
                           for n, e in sorted((emb.get("profiles") or {}).items())],
        "providers": list(PROVIDERS),
        "structured_modes": [s for s in STRUCTURED if s != "native"],
        "local_overrides": p.local_yaml.is_file(),
    }


# ---------------------------------------------------------------- probe
async def probe_profile(profile: str, override_fields: dict[str, Any] | None = None, *,
                        api_key: str | None = None, paths: ConfigPaths | None = None,
                        transport: Any = None, timeout_s: float = 30.0) -> dict[str, Any]:
    """One tiny structured call (``{"ok": true}``) against a saved or unsaved profile.

    ``api_key`` (write-only form value) is used for this probe only and never stored.
    """
    from .factory import build_llm, load_dotenv
    from .protocol import LLMRequest, extract_json

    p = paths or ConfigPaths()
    load_dotenv(p.env_file)
    cfg = load_merged_llm_config(p)
    secrets = known_secrets(cfg) + ([api_key] if api_key else [])
    base = (cfg.get("profiles") or {}).get(profile)
    result: dict[str, Any] = {"ok": False, "latency_ms": None, "provider": None, "model": None,
                              "error": None, "profile": profile}
    llm = None
    try:
        embed_names = sorted((cfg.get("embeddings") or {}).get("profiles") or {})
        clean = validate_fields(override_fields or {}, embed_profiles=embed_names,
                                existing=base)
        prof = {**(base or {}), **clean}
        result["provider"] = prof.get("provider")
        result["model"] = prof.get("fast_model") or prof.get("model")
        llm = build_llm(profile, prof, transport, api_key=api_key)
        req = LLMRequest(purpose="probe", agent_id="probe", tick=0,
                         shared_prefix="Connection test.",
                         call_head='Reply with exactly this JSON object: {"ok": true}',
                         user_msg='{"ok": true}', response_schema=PROBE_SCHEMA)
        t0 = time.perf_counter()
        limit = min(float(prof.get("timeout_s") or timeout_s), timeout_s)
        resp = await asyncio.wait_for(llm.complete_json(req), limit)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["model"] = resp.model or result["model"]
        data = extract_json(resp.text)
        if data.get("ok") is not True:
            raise ConfigError('model replied, but not with {"ok": true}')
        result["ok"] = True
    except TimeoutError:
        result["error"] = "timed out"
    except Exception as exc:  # report, never raise: this backs a UI button
        result["error"] = scrub(f"{type(exc).__name__}: {exc}", secrets)[:300]
    finally:
        aclose = getattr(llm, "aclose", None)
        if aclose is not None:
            await aclose()
    return result

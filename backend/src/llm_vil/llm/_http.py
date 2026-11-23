"""Shared async HTTP plumbing for provider adapters (httpx)."""

from __future__ import annotations

import os
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .protocol import LLMRequest, ProviderError, dumps

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def public_url(url: str) -> str:
    """Drop ``user:pass@`` userinfo so credentials in a base_url never reach stats/UI."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


# Only variables that are clearly LLM API keys may be named by a profile: a profile must
# never be able to read (and send to an arbitrary base_url) or overwrite PATH, LD_PRELOAD,
# cloud / VCS / payment credentials, proxies, etc.
LLM_KEY_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,47}_API_KEY$")
BLOCKED_ENV_PREFIXES = (
    "AWS_", "AZURE_", "GCP_", "GOOGLE_APPLICATION", "GITHUB_", "GITLAB_", "BITBUCKET_",
    "NPM_", "PYPI_", "SSH_", "GPG_", "STRIPE_", "PAYPAL_", "SLACK_", "TWILIO_", "SENDGRID_",
    "DATABASE_", "DB_", "POSTGRES", "MYSQL_", "REDIS_", "MONGO", "LD_", "DYLD_", "PYTHON",
    "PATH", "HOME", "SHELL", "DOCKER_", "KUBE", "VAULT_", "HEROKU_", "VERCEL_", "NETLIFY_",
    "CLOUDFLARE_", "DIGITALOCEAN_", "TERRAFORM_", "ANSIBLE_", "JWT_", "SECRET_", "SESSION_",
)


def is_llm_key_env(name: str | None) -> bool:
    """Strict allowlist for ``api_key_env``: ``*_API_KEY`` and nothing sensitive."""
    if not name or not LLM_KEY_ENV.match(name):
        return False
    if name.startswith(BLOCKED_ENV_PREFIXES) or name.endswith("_PROXY") or "PROXY" in name:
        return False
    return True


def resolve_api_key(api_key_env: str | None, required: bool) -> str | None:
    """API keys come only from the environment (``.env`` is loaded into it by the factory)."""
    if not api_key_env:
        return None
    if not is_llm_key_env(api_key_env):
        raise ProviderError("api_key_env must be an LLM key variable like MYPROVIDER_API_KEY",
                            retryable=False)
    key = os.environ.get(api_key_env)
    if required and not key:
        raise ProviderError(f"environment variable {api_key_env} is not set", retryable=False)
    return key or None


class HttpAdapter:
    """Base: owns an ``httpx.AsyncClient``; maps HTTP failures to scrubbed ProviderErrors."""

    provider = "http"

    def __init__(self, *, base_url: str, fast_model: str, smart_model: str | None = None,
                 api_key: str | None = None, extra_headers: dict[str, str] | None = None,
                 timeout_s: float = 60.0, max_tokens: int = 1024,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.fast_model = fast_model
        self.smart_model = smart_model or fast_model
        self.model = fast_model
        self._api_key = api_key  # never logged, returned, or put in error text
        self.extra_headers = dict(extra_headers or {})
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    def __repr__(self) -> str:  # keep keys out of reprs / tracebacks
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.fast_model!r})"

    def model_for(self, req: LLMRequest) -> str:
        return self.smart_model if req.tier == "deep" else self.fast_model

    def scrub(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text

    async def post_json(self, url: str, body: dict[str, Any],
                        headers: dict[str, str]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        merged = httpx.Headers({"content-type": "application/json"})
        merged.update(self.extra_headers)
        merged.update(headers)  # case-insensitive: adapter headers win over extras
        try:
            resp = await self._client.post(url, content=dumps(body).encode("utf-8"),
                                           headers=merged)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.provider} request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.scrub(f"{self.provider} transport error: "
                                           f"{type(exc).__name__}"), retryable=True) from exc
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            retry_after = None
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    retry_after = None
            raise ProviderError(
                self.scrub(f"{self.provider} HTTP {resp.status_code}: {resp.text[:300]}"),
                status=resp.status_code, retryable=resp.status_code in RETRYABLE_STATUS,
                retry_after=retry_after,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.provider} returned non-JSON body",
                                retryable=True) from exc
        return data, latency

    def describe(self) -> dict[str, Any]:
        """Public, secret-free profile description for /api/llm/stats."""
        return {"provider": self.provider, "base_url": public_url(self.base_url),
                "fast_model": self.fast_model, "smart_model": self.smart_model,
                "extra_header_names": sorted(self.extra_headers)}

    async def aclose(self) -> None:
        await self._client.aclose()

"""Anthropic Messages API (raw HTTP) with tool-call structured output.

* Models that accept it get forced ``tool_choice: {type: tool, name: respond}``.
* Claude Opus 5.5, Sonnet 5.5 and Fable/Mythos 5.1+ reject forced tool choice with a 400,
  so they get ``tool_choice: auto`` plus an explicit "call respond" instruction; any text
  reply is still parsed (and repaired) by the router.
* The shared prefix and the per-call head + persona are separate system blocks marked
  ``cache_control: ephemeral`` so prompt caching reuses them across villagers and ticks.
* ``stop_reason: refusal`` becomes a non-retryable ProviderError (router falls back).
"""

from __future__ import annotations

from typing import Any

from ._http import HttpAdapter
from .protocol import LLMRequest, LLMResponse, ProviderError, dumps

ANTHROPIC_VERSION = "2023-06-01"
TOOL_NAME = "respond"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Model families that 400 on forced tool_choice (any/tool).
NO_FORCED_TOOL = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1",
                  "claude-fable-5-2", "claude-mythos-5-1", "claude-mythos-5-2")
# Models that accept server-side refusal fallbacks (``fallbacks: "default"``).
SUPPORTS_FALLBACKS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                      "claude-fable-5-1", "claude-fable-5-2")


def forced_tool_ok(model: str) -> bool:
    return not model.startswith(NO_FORCED_TOOL)


class AnthropicAdapter(HttpAdapter):
    provider = "anthropic"

    def __init__(self, *, effort_fast: str | None = "low", effort_smart: str | None = "medium",
                 refusal_fallbacks: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("max_tokens", 16000)
        super().__init__(**kwargs)
        self.effort_fast = effort_fast
        self.effort_smart = effort_smart
        self.refusal_fallbacks = refusal_fallbacks

    def build_body(self, req: LLMRequest) -> dict[str, Any]:
        system: list[dict[str, Any]] = []
        if req.shared_prefix:
            system.append({"type": "text", "text": req.shared_prefix,
                           "cache_control": {"type": "ephemeral"}})
        head = "\n\n".join(p for p in (req.call_head, req.persona_core) if p)
        if head:
            system.append({"type": "text", "text": head, "cache_control": {"type": "ephemeral"}})
        user = req.user_msg
        if req.feedback:
            user = f"{user}\n\n[이전 응답 오류] {req.feedback}"
        schema = req.response_schema or {"type": "object", "properties": {}}
        model = self.model_for(req)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": TOOL_NAME, "description": f"Structured reply for {req.purpose}",
                       "input_schema": schema}],
        }
        if forced_tool_ok(model):
            body["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
        else:
            body["tool_choice"] = {"type": "auto"}
            body["messages"][0]["content"] = (
                f"{user}\n\nAnswer by calling the `{TOOL_NAME}` tool exactly once.")
        effort = self.effort_smart if req.tier == "deep" else self.effort_fast
        if effort:
            body["output_config"] = {"effort": effort}
        if self.refusal_fallbacks and model.startswith(SUPPORTS_FALLBACKS):
            body["fallbacks"] = "default"
        return body

    async def complete_json(self, req: LLMRequest) -> LLMResponse:
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        body = self.build_body(req)
        if "fallbacks" in body:
            # merge with any betas the profile already sends via extra_headers
            existing = {k.lower(): v for k, v in self.extra_headers.items()}.get(
                "anthropic-beta", "")
            betas = [b.strip() for b in existing.split(",") if b.strip()]
            if FALLBACK_BETA not in betas:
                betas.append(FALLBACK_BETA)
            headers["anthropic-beta"] = ",".join(betas)
        data, latency = await self.post_json(f"{self.base_url}/v1/messages", body, headers)
        if data.get("stop_reason") == "refusal":
            category = (data.get("stop_details") or {}).get("category")
            raise ProviderError(f"anthropic refused (category={category})", retryable=False)
        text = ""
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
                text = dumps(block.get("input") or {})
                break
            if block.get("type") == "text" and not text:
                text = block.get("text") or ""
        usage = data.get("usage") or {}
        cached = int(usage.get("cache_read_input_tokens") or 0)
        fresh = int(usage.get("input_tokens") or 0)
        created = int(usage.get("cache_creation_input_tokens") or 0)
        return LLMResponse(text=text, model=data.get("model") or self.model_for(req),
                           tokens_in=fresh + cached + created,
                           tokens_out=int(usage.get("output_tokens") or 0),
                           tokens_cached=cached, latency_ms=latency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic has no embeddings API; use a local embedder")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "structured": "tool_call",
                "forced_tool_choice": forced_tool_ok(self.fast_model)}

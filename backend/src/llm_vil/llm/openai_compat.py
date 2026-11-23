"""OpenAI-compatible chat completions (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, LiteLLM).

``structured`` modes:
  * ``json_schema``  -> ``response_format: {type: json_schema, json_schema: {...}}``
  * ``json_object``  -> ``response_format: {type: json_object}`` + schema in the system prompt
  * ``prompt_only``  -> schema in the system prompt only; ``<think>`` blocks and prose around
                        the outermost JSON object are stripped by the router's ``extract_json``
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ._http import HttpAdapter
from .protocol import LLMRequest, LLMResponse, dumps

Structured = Literal["json_schema", "json_object", "prompt_only"]


def schema_instruction(schema: dict[str, Any]) -> str:
    return ("Respond with ONE JSON object only (no prose, no code fences) matching this JSON "
            f"schema:\n{dumps(schema)}")


class OpenAICompatAdapter(HttpAdapter):
    provider = "openai_compat"

    def __init__(self, *, structured: Structured = "json_schema", temperature: float = 0.7,
                 base_url: str, fast_model: str, smart_model: str | None = None,
                 api_key: str | None = None, extra_headers: dict[str, str] | None = None,
                 timeout_s: float = 60.0, max_tokens: int = 1024,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__(base_url=base_url, fast_model=fast_model, smart_model=smart_model,
                         api_key=api_key, extra_headers=extra_headers, timeout_s=timeout_s,
                         max_tokens=max_tokens, transport=transport)
        self.structured: Structured = structured
        self.temperature = temperature

    def build_body(self, req: LLMRequest) -> dict[str, Any]:
        system = "\n\n".join(p for p in (req.shared_prefix, req.call_head, req.persona_core) if p)
        if self.structured != "json_schema" and req.response_schema:
            system = f"{system}\n\n{schema_instruction(req.response_schema)}"
        user = req.user_msg
        if req.feedback:
            user = f"{user}\n\n[이전 응답 오류] {req.feedback}"
        body: dict[str, Any] = {
            "model": self.model_for(req),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.structured == "json_schema" and req.response_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": req.purpose, "schema": req.response_schema, "strict": False}}
        elif self.structured == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def complete_json(self, req: LLMRequest) -> LLMResponse:
        headers = {"authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        data, latency = await self.post_json(f"{self.base_url}/chat/completions",
                                             self.build_body(req), headers)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return LLMResponse(text=text, model=data.get("model") or self.model_for(req),
                           tokens_in=int(usage.get("prompt_tokens") or 0),
                           tokens_out=int(usage.get("completion_tokens") or 0),
                           tokens_cached=int(cached), latency_ms=latency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("use an Embedder (embeddings profile) for vectors")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "structured": self.structured}

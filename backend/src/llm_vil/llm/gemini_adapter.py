"""Gemini ``generateContent`` with ``responseSchema`` (flat OpenAPI subset)."""

from __future__ import annotations

from typing import Any

from ._http import HttpAdapter
from .protocol import LLMRequest, LLMResponse

ALLOWED_KEYS = frozenset({"type", "format", "description", "nullable", "enum", "properties",
                          "required", "items", "minItems", "maxItems", "minimum", "maximum"})


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``, drop unsupported keys (``additionalProperties``, ``default``,
    ``title`` ...) and turn ``anyOf[X, null]`` into ``nullable`` X. Never emits unions."""
    defs = schema.get("$defs", {})

    def conv(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return conv(defs[node["$ref"].rsplit("/", 1)[-1]])
        if "anyOf" in node or "oneOf" in node:
            options = [o for o in node.get("anyOf", node.get("oneOf", []))
                       if o.get("type") != "null"]
            out = conv(options[0]) if options else {"type": "STRING"}
            if len(options) < len(node.get("anyOf", node.get("oneOf", []))):
                out["nullable"] = True
            return out
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k not in ALLOWED_KEYS:
                continue
            if k == "type":
                out[k] = str(v).upper()
            elif k == "properties":
                out[k] = {name: conv(sub) for name, sub in v.items()}
            elif k == "items":
                out[k] = conv(v)
            elif k == "enum":
                out[k] = [str(x) for x in v]
            else:
                out[k] = v
        if "enum" in out and out.get("type") not in (None, "STRING"):
            out.pop("enum")  # Gemini enums are string-only
        return out

    return conv(schema)


class GeminiAdapter(HttpAdapter):
    provider = "gemini"

    def build_body(self, req: LLMRequest) -> dict[str, Any]:
        system = "\n\n".join(p for p in (req.shared_prefix, req.call_head, req.persona_core) if p)
        user = req.user_msg
        if req.feedback:
            user = f"{user}\n\n[이전 응답 오류] {req.feedback}"
        gen: dict[str, Any] = {"responseMimeType": "application/json",
                               "maxOutputTokens": self.max_tokens}
        if req.response_schema:
            gen["responseSchema"] = to_gemini_schema(req.response_schema)
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        if system:  # an empty systemInstruction is rejected by the API
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    async def complete_json(self, req: LLMRequest) -> LLMResponse:
        model = self.model_for(req)
        headers = {"x-goog-api-key": self._api_key} if self._api_key else {}  # never in URL
        data, latency = await self.post_json(
            f"{self.base_url}/v1beta/models/{model}:generateContent", self.build_body(req),
            headers)
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        # Gemini 2.5 "thinking" parts are flagged thought=true: never part of the answer
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata") or {}
        return LLMResponse(text=text, model=data.get("modelVersion") or model,
                           tokens_in=int(usage.get("promptTokenCount") or 0),
                           tokens_out=int(usage.get("candidatesTokenCount") or 0),
                           tokens_cached=int(usage.get("cachedContentTokenCount") or 0),
                           latency_ms=latency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("use an Embedder (embeddings profile) for vectors")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "structured": "json_schema"}

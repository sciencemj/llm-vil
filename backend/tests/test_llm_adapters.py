"""Provider adapters against httpx.MockTransport: request shapes, parsing, backoff, secrets."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from llm_vil.agent.replies import RoutineReply, TurnReply, flat_schema
from llm_vil.llm.anthropic_adapter import AnthropicAdapter, forced_tool_ok
from llm_vil.llm.factory import build_bundle, load_dotenv
from llm_vil.llm.gemini_adapter import GeminiAdapter, to_gemini_schema
from llm_vil.llm.openai_compat import OpenAICompatAdapter
from llm_vil.llm.protocol import LLMRequest, LLMRouter, ProviderError, dumps

SECRET = "sk-test-SUPERSECRET-1234567890"


def req(tier: str = "fast", schema: dict | None = None) -> LLMRequest:
    return LLMRequest(purpose="choose_routine", agent_id="mina", tick=500, tier=tier,
                      shared_prefix="마을 규칙", call_head="할 일을 골라라", persona_core="민아",
                      user_msg='{"needs": {}}',
                      response_schema=schema or flat_schema(RoutineReply))


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses.pop(0)
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def chat_reply(content: str, cached: int = 0) -> dict:
    return {"model": "gpt-test", "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def oa(rec: Recorder, structured: str = "json_schema", **kw) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(base_url="https://llm.example.com/v1/", fast_model="fast-m",
                               smart_model="smart-m", api_key=SECRET, structured=structured,
                               extra_headers={"HTTP-Referer": "http://127.0.0.1:5174",
                                              "X-Title": "llm-vil"},
                               transport=httpx.MockTransport(rec), **kw)


# ---------------------------------------------------------------- openai_compat
async def test_openai_json_schema_mode_custom_base_url_and_headers():
    rec = Recorder([chat_reply('{"routine": "rest", "target": "home_mina"}', cached=100)])
    adapter = oa(rec)
    resp = await adapter.complete_json(req("deep"))
    r = rec.requests[0]
    assert str(r.url) == "https://llm.example.com/v1/chat/completions"
    assert r.headers["authorization"] == f"Bearer {SECRET}"
    assert r.headers["x-title"] == "llm-vil" and r.headers["http-referer"]
    body = rec.body()
    assert body["model"] == "smart-m"  # deep tier -> smart model
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "choose_routine"
    assert "$ref" not in dumps(rf) and "anyOf" not in dumps(rf)
    assert resp.tokens_in == 120 and resp.tokens_cached == 100
    assert "마을 규칙" in body["messages"][0]["content"]  # Korean kept (ensure_ascii=False)
    assert "\\u" not in r.content.decode()


async def test_openai_json_object_and_prompt_only_put_schema_in_system():
    for mode in ("json_object", "prompt_only"):
        rec = Recorder([chat_reply('{"routine": "rest"}')])
        await oa(rec, structured=mode).complete_json(req())
        body = rec.body()
        assert body["model"] == "fast-m"
        assert "JSON schema" in body["messages"][0]["content"]
        if mode == "json_object":
            assert body["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in body


async def test_prompt_only_think_stripping_and_repair_retry():
    rec = Recorder([
        chat_reply("<think>hmm {\"x\":1}</think> 여기요: {\"routine\": \"fly\"}"),  # invalid enum
        chat_reply("<think>다시</think>```json\n{\"routine\": \"rest\"}\n```"),
    ])
    router = LLMRouter(oa(rec, structured="prompt_only"), retry_base_delay_s=0)
    out = await router.call(req(), RoutineReply)
    assert out is not None and out.routine == "rest"
    assert router.stats.repaired == 1 and router.obs.recent[-1].status == "repaired"
    assert "[이전 응답 오류]" in rec.body(1)["messages"][1]["content"]


async def test_429_exponential_backoff_with_retry_after():
    rec = Recorder([
        httpx.Response(429, headers={"retry-after": "0.05"}, json={"error": "slow down"}),
        httpx.Response(503, json={"error": "busy"}),
        chat_reply('{"routine": "rest"}'),
    ])
    router = LLMRouter(oa(rec), retry_base_delay_s=0.01)
    t0 = time.perf_counter()
    out = await router.call(req(), RoutineReply)
    assert out is not None and len(rec.requests) == 3
    assert time.perf_counter() - t0 >= 0.05 + 0.02  # retry-after, then 2x base backoff
    assert router.stats.errors == 2


async def test_auth_error_is_not_retried_and_never_leaks_key():
    rec = Recorder([httpx.Response(401, json={"error": f"bad key {SECRET}"})])
    router = LLMRouter(oa(rec), retry_base_delay_s=0)
    assert await router.call(req(), RoutineReply) is None
    assert len(rec.requests) == 1  # non-retryable
    blob = dumps([t.model_dump() for t in router.drain_tape()]) + dumps(
        [r.model_dump() for r in router.obs.recent])
    assert SECRET not in blob and "***" in blob
    assert SECRET not in repr(oa(Recorder([])))


# ---------------------------------------------------------------- anthropic
def anthropic_reply(tool_input: dict, stop: str = "tool_use") -> dict:
    return {"model": "claude-haiku-5-5", "stop_reason": stop,
            "content": [{"type": "thinking", "thinking": ""},
                        {"type": "tool_use", "name": "respond", "input": tool_input}],
            "usage": {"input_tokens": 40, "cache_read_input_tokens": 300,
                      "cache_creation_input_tokens": 0, "output_tokens": 25}}


def ant(rec: Recorder, fast: str) -> AnthropicAdapter:
    return AnthropicAdapter(base_url="https://api.anthropic.com", fast_model=fast,
                            smart_model="claude-opus-5-5", api_key=SECRET,
                            transport=httpx.MockTransport(rec))


async def test_anthropic_forced_tool_call_and_cache_control():
    rec = Recorder([anthropic_reply({"routine": "rest", "target": "home_mina"})])
    resp = await ant(rec, "claude-haiku-5-5").complete_json(req())
    r = rec.requests[0]
    assert str(r.url) == "https://api.anthropic.com/v1/messages"
    assert r.headers["x-api-key"] == SECRET and r.headers["anthropic-version"] == "2023-06-01"
    body = rec.body()
    assert body["tool_choice"] == {"type": "tool", "name": "respond"}
    assert body["tools"][0]["input_schema"]["properties"]["routine"]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in body["system"])
    assert body["output_config"] == {"effort": "low"}
    assert json.loads(resp.text)["routine"] == "rest"
    assert resp.tokens_cached == 300 and resp.tokens_in == 340


async def test_anthropic_models_without_forced_tool_choice_use_auto():
    assert not forced_tool_ok("claude-opus-5-5") and forced_tool_ok("claude-haiku-5-5")
    rec = Recorder([anthropic_reply({"routine": "rest"})])
    await ant(rec, "claude-opus-5-5").complete_json(req())
    body = rec.body()
    assert body["tool_choice"] == {"type": "auto"}
    assert "respond" in body["messages"][0]["content"]
    assert body["fallbacks"] == "default"
    assert rec.requests[0].headers["anthropic-beta"] == "server-side-fallback-2026-07-01"


async def test_anthropic_refusal_is_a_non_retryable_fallback():
    rec = Recorder([{**anthropic_reply({}), "stop_reason": "refusal",
                     "stop_details": {"category": "cyber"}}])
    router = LLMRouter(ant(rec, "claude-haiku-5-5"), retry_base_delay_s=0)
    assert await router.call(req(), RoutineReply) is None
    assert len(rec.requests) == 1 and router.obs.recent[-1].status == "fallback"


# ---------------------------------------------------------------- gemini
def test_gemini_schema_is_flat_without_refs_or_unions():
    nested = {
        "type": "object", "additionalProperties": False, "title": "X",
        "$defs": {"Item": {"type": "object", "properties": {"t": {"type": "string",
                                                                    "default": "a"}}}},
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
            "maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "kind": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["items"],
    }
    flat = to_gemini_schema(nested)
    text = dumps(flat)
    for bad in ("$ref", "$defs", "anyOf", "additionalProperties", "default", "title"):
        assert bad not in text
    assert flat["properties"]["maybe"] == {"type": "INTEGER", "nullable": True}
    assert flat["properties"]["items"]["items"]["properties"]["t"]["type"] == "STRING"
    assert flat["properties"]["kind"]["enum"] == ["a", "b"]
    assert "$ref" not in dumps(to_gemini_schema(flat_schema(TurnReply)))


async def test_gemini_request_uses_header_key_and_response_schema():
    rec = Recorder([{"candidates": [{"content": {"parts": [{"text": '{"routine": "rest"}'}]}}],
                     "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 9,
                                       "cachedContentTokenCount": 10},
                     "modelVersion": "gemini-test"}])
    adapter = GeminiAdapter(base_url="https://generativelanguage.googleapis.com",
                            fast_model="gemini-2.5-flash", api_key=SECRET,
                            transport=httpx.MockTransport(rec))
    resp = await adapter.complete_json(req())
    r = rec.requests[0]
    assert SECRET not in str(r.url)  # key travels in a header, never the URL
    assert r.headers["x-goog-api-key"] == SECRET
    assert r.url.path.endswith("/models/gemini-2.5-flash:generateContent")
    gen = rec.body()["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "$ref" not in dumps(gen["responseSchema"]) and "anyOf" not in dumps(gen)
    assert resp.tokens_cached == 10 and json.loads(resp.text) == {"routine": "rest"}


# ---------------------------------------------------------------- factory / secrets
def test_factory_reads_keys_from_env_file_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# local secrets\nANTHROPIC_API_KEY={SECRET}\n", encoding="utf-8")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        build_bundle("anthropic", "demo", env_file=tmp_path / "missing.env")
    assert load_dotenv(env) == ["ANTHROPIC_API_KEY"]
    bundle = build_bundle("anthropic", "demo", env_file=env)
    info = dumps(bundle.public_info())
    assert SECRET not in info and "claude-opus-5-5" in info
    assert bundle.pricing["claude-opus-5-5"].output == 20.0


def test_unknown_profiles_are_rejected():
    with pytest.raises(ValueError, match="unknown LLM profile"):
        build_bundle("nope")
    with pytest.raises(ValueError, match="embeddings profile"):
        build_bundle("demo", "nope")


async def test_cost_estimate_uses_per_model_pricing():
    from llm_vil.llm.protocol import Pricing

    rec = Recorder([anthropic_reply({"routine": "rest"})])
    router = LLMRouter(ant(rec, "claude-haiku-5-5"), retry_base_delay_s=0,
                       pricing={"claude-haiku-5-5": Pricing(input=1.0, output=5.0,
                                                            cached_input=0.1)})
    assert await router.call(req(), RoutineReply) is not None
    # 40 fresh in * 1 + 300 cached * 0.1 + 25 out * 5  (per 1M)
    assert router.obs.cost_usd == pytest.approx((40 * 1 + 300 * 0.1 + 25 * 5) / 1e6)

"""Regression tests for the M3 review (5 Major + 6 Minor)."""

from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_vil.agent.memory import MemoryStream
from llm_vil.agent.replies import RoutineReply
from llm_vil.engine import SaveStore, SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm._http import public_url
from llm_vil.llm.anthropic_adapter import AnthropicAdapter
from llm_vil.llm.embeddings import (
    CachingEmbedder,
    EmbedderUnavailable,
    OpenAICompatEmbedder,
    reembed_missing,
)
from llm_vil.llm.factory import build_bundle, load_dotenv
from llm_vil.llm.gemini_adapter import GeminiAdapter
from llm_vil.llm.protocol import (
    HashingEmbedder,
    LLMRequest,
    LLMRouter,
    TokenBudget,
    dumps,
)
from llm_vil.schemas.config import MemoryConfig
from llm_vil.server import create_app

SECRET = "sk-review-SECRET-987654321"


def req(**kw) -> LLMRequest:
    base = {"purpose": "choose_routine", "agent_id": "mina", "tick": 400, "user_msg": "{}"}
    return LLMRequest(**{**base, **kw})


# ---------------------------------------------------------------- Major 1: timeouts/budget
def test_profile_timeout_and_runtime_settings_reach_the_router(config, tmp_path, monkeypatch):
    assert build_bundle("ollama_local", "demo").llm_timeout == 120.0
    assert build_bundle("demo").llm_timeout == 30.0  # no profile timeout: runtime default
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "default_profile: demo\nprofiles:\n  demo: {provider: demo, model: d, embeddings: demo}\n"
        "embeddings: {default: demo, profiles: {demo: {provider: demo, dim: 64}}}\n"
        "runtime: {json_repair_retries: 1, token_budget_per_day: 500, "
        "token_budget_per_run: 9000, timeout_seconds: 7}\n", encoding="utf-8")
    app = create_app(config, paused=True, llm_config_path=str(cfg))
    with TestClient(app):
        router = app.state.runtime.engine.router
        assert router.retries == 1 and router.timeout == 7.0
        assert router.budget.per_day == 500 and router.budget.per_run == 9000
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    app2 = create_app(config, paused=True, llm_profile="openai_compat", embed_profile="demo")
    with TestClient(app2):
        assert app2.state.runtime.engine.router.timeout == 60.0  # profile timeout_s wins


# ---------------------------------------------------------------- Major 2: Gemini thoughts
async def test_gemini_thought_parts_are_ignored():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "systemInstruction" not in body  # empty system prompt is omitted
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": 'thinking... {"routine": "sleep"} maybe?', "thought": True},
            {"text": '{"routine": "rest"}'},
        ]}}], "usageMetadata": {}})

    g = GeminiAdapter(base_url="https://g.example", fast_model="gemini-2.5-flash",
                      api_key=SECRET, transport=httpx.MockTransport(handler))
    out = await LLMRouter(g, retry_base_delay_s=0).call(req(), RoutineReply)
    assert out is not None and out.routine == "rest"


# ---------------------------------------------------------------- Major 3: off-loop ONNX
class SlowBlockingEmbedder(CachingEmbedder):
    provider = "test"
    blocking = True

    def __init__(self):
        super().__init__("slow-test", 16)
        self.threads: set[int] = set()

    def _compute(self, texts):
        self.threads.add(threading.get_ident())
        return [[float((len(t) + i) % 7 + 1) for i in range(self.dim)] for t in texts]

    def embed(self, text):  # the sync path must never be used for blocking embedders
        raise AssertionError("blocking embedder called on the event loop")


async def test_blocking_embedder_never_runs_on_the_event_loop(config):
    emb = SlowBlockingEmbedder()
    e = SimEngine(config, DemoBackend(), embedder=emb)
    loop_thread = threading.get_ident()
    await e.run_until(config.start_minute + 240)
    assert emb.threads and loop_thread not in emb.threads
    mem = e.agents["mina"].memory
    assert mem.records and all(r.embed_model == "slow-test" for r in mem.records)
    hits = await mem.aretrieve(e.tick, "밭", k=3)
    assert hits and e.metrics["errors"] == 0


# ---------------------------------------------------------------- Major 4: lifecycle/scrub
def test_openai_embedder_scrubs_transport_errors_and_closes():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused, auth={request.headers['authorization']}")

    emb = OpenAICompatEmbedder(base_url="https://e.example/v1", model="m", dim=4,
                               api_key=SECRET, transport=httpx.MockTransport(boom))
    with pytest.raises(EmbedderUnavailable) as err:
        emb.embed("안녕")
    assert SECRET not in str(err.value) and err.value.__cause__ is None
    emb.close()
    emb.close()  # idempotent


def test_failed_startup_still_closes_store(config, tmp_path, monkeypatch):
    import llm_vil.server.app as app_mod

    closed: list[str] = []

    class TrackingStore(SaveStore):
        def close(self) -> None:
            closed.append(self.path)
            super().close()

    monkeypatch.setattr(app_mod, "SaveStore", TrackingStore)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(config, paused=True, store_path=str(tmp_path / "s.sqlite"),
                     llm_profile="anthropic", env_file=str(tmp_path / "none.env"))
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"), TestClient(app):
        pass
    assert closed == [str(tmp_path / "s.sqlite")]


def test_explicit_llm_skips_building_the_profile_adapter(config, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(config, DemoBackend(), paused=True, llm_profile="anthropic",
                     embed_profile="demo", env_file=str(tmp_path / "none.env"))
    with TestClient(app) as c:  # would raise if the keyless Anthropic adapter were built
        assert c.get("/api/health").json()["ok"]


# ---------------------------------------------------------------- Major 5: keyset reembed
def test_reembed_is_keyset_paginated_and_transactional(tmp_path):
    store = SaveStore(tmp_path / "k.sqlite")
    old = MemoryStream("mina", HashingEmbedder(), MemoryConfig())
    recs = [old.add(i, "observation", f"기억 {i}", 3, subjects=["junho"]) for i in range(23)]
    store.write_batch(archive=[(r, 10) for r in recs])

    class NewEmb(CachingEmbedder):
        provider = "t"

        def __init__(self):
            super().__init__("new-model", 8)
            self.batches: list[int] = []

        def _compute(self, texts):
            self.batches.append(len(texts))
            return [[1.0 + i for i in range(8)] for _ in texts]

    emb = NewEmb()
    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT rowid FROM memory_archive WHERE rowid > 0 AND "
        "(embed_model != 'x' OR embed_dim != 8) ORDER BY rowid LIMIT 5").fetchall()
    assert "SCAN memory_archive" not in str(plan) or "rowid>?" in str(plan).replace(" ", "")
    assert reembed_missing(store, emb, batch=5) == 23
    assert emb.batches == [5, 5, 5, 5, 3]  # each row visited once
    assert reembed_missing(store, emb, batch=5) == 0
    # a failing batch rolls back completely
    real = store.conn

    class Failing:
        def execute(self, sql, *a):
            if sql.startswith("UPDATE memory_archive"):
                raise RuntimeError("disk full")
            return real.execute(sql, *a)

    store.conn = Failing()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        store.update_archive_embeddings("other", [("mina", recs[0].id, [0.0] * 8)])
    store.conn = real
    assert not real.in_transaction
    assert store.archive_needing_embedding("new-model", 8) == []


# ---------------------------------------------------------------- minors
def test_dotenv_quotes_and_inline_comments(tmp_path, monkeypatch):
    for k in ("A_KEY", "B_KEY", "C_KEY", "D_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text('A_KEY="has # hash"\nB_KEY=plain # comment\nC_KEY=\'q\'"\nD_KEY="x\n',
                   encoding="utf-8")
    load_dotenv(env)
    import os

    assert os.environ["A_KEY"] == "has # hash"
    assert os.environ["B_KEY"] == "plain"
    assert os.environ["C_KEY"] == "'q'\""  # mismatched quotes are kept verbatim
    assert os.environ["D_KEY"] == '"x'


def test_public_url_strips_userinfo():
    assert public_url("https://user:pw@llm.example:8443/v1") == "https://llm.example:8443/v1"
    assert public_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_lookup_dedupes_duplicate_misses():
    class Count(CachingEmbedder):
        provider = "t"

        def __init__(self):
            super().__init__("c", 4)
            self.seen: list[str] = []

        def _compute(self, texts):
            self.seen.extend(texts)
            return [[1.0, 2.0, 3.0, 4.0] for _ in texts]

    c = Count()
    out = c.embed_many(["가", "나", "가", "가"])
    assert c.seen == ["가", "나"] and out[0] == out[2] == out[3]


def test_mismatched_vectors_do_not_outrank_dissimilar_ones():
    emb = HashingEmbedder()
    m = MemoryStream("mina", emb, MemoryConfig())
    qv = emb.embed("장작 패기")
    base = m.add(0, "observation", "장작 패기", 3)
    similar = base.model_copy(update={"id": "b", "embedding": qv})
    dissimilar = base.model_copy(update={"id": "c", "embedding": [-x for x in qv]})  # cos -1
    foreign = base.model_copy(update={"id": "z", "embed_model": "other-model"})
    m.records = []
    ranked = m.retrieve(1, "장작 패기", extra=[foreign, dissimilar, similar], k=3)
    # old behaviour: foreign got relevance 0.0 > -1 and outranked the dissimilar memory
    assert [r.id for r in ranked] == ["b", "c", "z"]


async def test_budget_skip_is_visible_in_observability():
    router = LLMRouter(DemoBackend(), budget=TokenBudget(per_run=0))
    router.budget.paused = True
    assert await router.call(req(context={}), RoutineReply) is None
    assert router.obs.budget_skips == 1
    assert router.obs.recent[-1].status == "fallback" and "budget" in router.obs.recent[-1].error


async def test_anthropic_merges_profile_betas_with_fallback_beta():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "respond",
                                                      "input": {"routine": "rest"}}],
                                         "usage": {}, "stop_reason": "tool_use"})

    a = AnthropicAdapter(base_url="https://a.example", fast_model="claude-opus-5-5",
                         api_key=SECRET, extra_headers={"Anthropic-Beta": "my-beta-1"},
                         transport=httpx.MockTransport(handler))
    await a.complete_json(req())
    assert seen["beta"] == "my-beta-1,server-side-fallback-2026-07-01"


def test_stats_include_budget_skips(config):
    app = create_app(config, paused=True)
    with TestClient(app) as c:
        s = c.get("/api/llm/stats").json()
        assert "budget_skips" in s and SECRET not in dumps(s)


def test_no_event_loop_blocking_helper_is_async():
    assert asyncio.iscoroutinefunction(MemoryStream.aretrieve)

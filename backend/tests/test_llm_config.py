"""Provider & API-key configuration: store, CLI, REST endpoints, hot-swap."""

from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_vil import cli
from llm_vil.config import CONFIG_DIR
from llm_vil.engine import SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.config_store import (
    ConfigError,
    ConfigPaths,
    config_summary,
    load_merged_llm_config,
    mask_key,
    probe_profile,
    save_env_key,
    save_profile_override,
    scrub,
)
from llm_vil.llm.embeddings import DemoCachedEmbedder
from llm_vil.server import create_app
from llm_vil.server.runtime import SimRuntime

SECRET = "sk-cfg-TOPSECRET-55667788"


@pytest.fixture
def paths(tmp_path, monkeypatch) -> ConfigPaths:
    base = tmp_path / "llm.yaml"
    base.write_text((CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
                 "MYLLM_API_KEY"):
        monkeypatch.setenv(name, "x")  # records the original state so teardown restores it
        monkeypatch.delenv(name)
    return ConfigPaths(base, tmp_path / ".env")


# ---------------------------------------------------------------- store
def test_env_key_write_is_0600_atomic_and_preserves_other_lines(paths):
    paths.env_file.write_text("# keep me\nOTHER_VAR=1\nOPENAI_API_KEY=old\n", encoding="utf-8")
    save_env_key("OPENAI_API_KEY", SECRET, paths)
    text = paths.env_file.read_text(encoding="utf-8")
    assert "# keep me" in text and "OTHER_VAR=1" in text
    assert text.count("OPENAI_API_KEY=") == 1 and f"OPENAI_API_KEY={SECRET}" in text
    assert stat.S_IMODE(os.stat(paths.env_file).st_mode) == 0o600
    assert os.environ["OPENAI_API_KEY"] == SECRET
    assert not [p for p in paths.env_file.parent.iterdir() if p.name.startswith(".env.")]
    with pytest.raises(ConfigError):
        save_env_key("bad name", "x", paths)
    with pytest.raises(ConfigError):
        save_env_key("OPENAI_API_KEY", "two\nlines", paths)


def test_local_yaml_overrides_merge_over_base(paths):
    prof = save_profile_override("my_vllm", {
        "provider": "openai_compat", "base_url": "http://127.0.0.1:8000/v1",
        "fast_model": "qwen2.5-7b", "structured": "prompt_only", "api_key_env": "MYLLM_API_KEY",
        "timeout_s": "45", "embeddings": "local"}, set_default=True, paths=paths)
    assert prof["timeout_s"] == 45.0 and prof["structured"] == "prompt_only"
    save_profile_override("openai_compat", {"fast_model": "gpt-4.1-mini"}, paths=paths)
    merged = load_merged_llm_config(paths)
    assert merged["default_profile"] == "my_vllm"
    assert merged["profiles"]["openai_compat"]["fast_model"] == "gpt-4.1-mini"
    assert merged["profiles"]["openai_compat"]["smart_model"] == "gpt-4o"  # base kept
    local = yaml.safe_load(paths.local_yaml.read_text(encoding="utf-8"))
    assert "api_key" not in json.dumps(local).replace("api_key_env", "")
    base = yaml.safe_load(paths.base_yaml.read_text(encoding="utf-8"))
    assert base["default_profile"] == "demo"  # shipped file untouched
    for bad in ({"provider": "nope"}, {"structured": "xml"}, {"base_url": "ftp://x"},
                {"base_url": "https://u:p@host/v1"}, {"timeout_s": 9999},
                {"api_key": SECRET}, {"embeddings": "nope"}):
        with pytest.raises(ConfigError):
            save_profile_override("openai_compat", bad, paths=paths)
    with pytest.raises(ConfigError, match="provider"):
        save_profile_override("brand_new", {"fast_model": "x"}, paths=paths)


def test_summary_masks_keys(paths):
    save_env_key("OPENAI_API_KEY", SECRET, paths)
    s = config_summary("demo", "demo", paths)
    oa = next(p for p in s["profiles"] if p["name"] == "openai_compat")
    assert oa["has_api_key"] and oa["key_hint"] == "****7788"
    assert SECRET not in json.dumps(s)
    assert mask_key("short") == "****" and mask_key(None) is None
    assert scrub(f"Authorization: Bearer {SECRET} boom", [SECRET]) == \
        "Authorization: Bearer *** boom"


async def test_probe_demo_and_unsaved_form_without_leaks(paths):
    ok = await probe_profile("demo", paths=paths)
    assert ok["ok"] and ok["provider"] == "demo" and ok["latency_ms"] is not None
    bad = await probe_profile("custom", {"provider": "openai_compat",
                                         "base_url": "http://127.0.0.1:9/v1",
                                         "fast_model": "m", "api_key_env": "MYLLM_API_KEY"},
                              api_key=SECRET, paths=paths, timeout_s=3)
    assert not bad["ok"] and bad["error"] and SECRET not in json.dumps(bad)
    assert "MYLLM_API_KEY" not in os.environ  # an unsaved form key is never stored


# ---------------------------------------------------------------- CLI
def test_cli_show_set_test(paths, capsys, monkeypatch):
    argv = ["--config", str(paths.base_yaml), "--env-file", str(paths.env_file)]
    monkeypatch.setattr("getpass.getpass", lambda prompt="": SECRET)
    assert cli.main([*argv, "config", "set", "--profile", "my_vllm", "--provider",
                     "openai_compat", "--base-url", "http://127.0.0.1:8000/v1",
                     "--fast-model", "qwen", "--api-key-env", "MYLLM_API_KEY", "--prompt-key",
                     "--embed-profile", "demo", "--set-default"]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "mode 600" in out
    assert cli.main([*argv, "config", "show"]) == 0
    out = capsys.readouterr().out
    assert "* my_vllm" in out and "MYLLM_API_KEY ✓ ****7788" in out and SECRET not in out
    assert cli.main([*argv, "config", "test", "--profile", "demo"]) == 0
    assert "✓ demo" in capsys.readouterr().out
    assert cli.main([*argv, "config", "set", "--profile", "x", "--fast-model", "m"]) == 2


# ---------------------------------------------------------------- REST
@pytest.fixture
def client(config, paths):
    app = create_app(config, paused=True, llm_config_path=str(paths.base_yaml),
                     env_file=str(paths.env_file))
    with TestClient(app) as c:
        yield c, app


def test_get_config_never_returns_plaintext(client, paths):
    c, _ = client
    save_env_key("ANTHROPIC_API_KEY", SECRET, paths)
    r = c.get("/api/llm/config")
    assert r.status_code == 200 and SECRET not in r.text
    body = r.json()
    assert body["active_profile"] == "demo" and body["active_embed_profile"] == "demo"
    ant = next(p for p in body["profiles"] if p["name"] == "anthropic")
    assert ant["has_api_key"] and ant["key_hint"] == "****7788"
    assert c.get("/api/llm/config", headers={"origin": "https://evil.example"}).status_code \
        == 403


def test_post_config_saves_override_and_write_only_key(client, paths):
    c, _ = client
    r = c.post("/api/llm/config", json={
        "profile": "openrouter", "fields": {"fast_model": "meta/llama-3.1-8b"},
        "api_key": SECRET})
    assert r.status_code == 200 and SECRET not in r.text
    assert f"OPENROUTER_API_KEY={SECRET}" in paths.env_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(os.stat(paths.env_file).st_mode) == 0o600
    orp = next(p for p in r.json()["profiles"] if p["name"] == "openrouter")
    assert orp["fast_model"] == "meta/llama-3.1-8b" and orp["has_api_key"]
    bad = c.post("/api/llm/config", json={"profile": "openrouter",
                                          "fields": {"base_url": f"https://x:{SECRET}@h/v1"}})
    assert bad.status_code == 400 and SECRET not in bad.text
    assert c.post("/api/llm/config", json={"profile": "demo", "fields": {}},
                  headers={"origin": "http://evil.example"}).status_code == 403


def test_post_test_endpoint(client):
    c, _ = client
    ok = c.post("/api/llm/test", json={"profile": "demo"}).json()
    assert ok["ok"] and ok["provider"] == "demo"
    r = c.post("/api/llm/test", json={"profile": "anthropic", "api_key": SECRET,
                                      "fields": {"base_url": "http://127.0.0.1:9"}})
    assert r.status_code == 200 and not r.json()["ok"] and SECRET not in r.text


def test_switch_hot_swaps_under_lock_and_closes_old_client(client):
    c, app = client
    runtime = app.state.runtime
    closed = []

    class ClosingDemo(DemoBackend):
        async def aclose(self):
            closed.append(True)

    runtime.engine.router.llm = ClosingDemo()
    r = c.post("/api/llm/switch", json={"profile": "demo", "embed_profile": "demo"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["active_profile"] == "demo" and body["stats"]["provider"] == "demo"
    assert closed == [True]  # the old client was closed after the swap
    assert not isinstance(runtime.engine.router.llm, ClosingDemo)
    bad = c.post("/api/llm/switch", json={"profile": "anthropic"})  # no key configured
    assert bad.status_code == 400 and "ANTHROPIC_API_KEY" in bad.json()["detail"]
    assert c.post("/api/llm/switch", json={"profile": "demo"},
                  headers={"origin": "http://evil.example"}).status_code == 403


async def test_switch_cancels_inflight_cognition_and_swaps_embedder(config):
    class Gated(DemoBackend):
        def __init__(self):
            super().__init__()
            self.gate = asyncio.Event()
            self.closed = False

        async def complete_json(self, req):
            await self.gate.wait()
            return await super().complete_json(req)

        async def aclose(self):
            self.closed = True

    from llm_vil.engine import CognitionClock
    from llm_vil.schemas.world import AtomicAction

    old = Gated()
    e = SimEngine(config, old, clock=CognitionClock(t_max_seconds=60.0))
    rt = SimRuntime(e, paused=True)
    e.world.apply("mina", AtomicAction(kind="wake"))
    e._spawn("mina", "choose_routine", "test")
    await asyncio.sleep(0)
    assert e.jobs and e.router.gate.in_flight == 1
    new_emb = DemoCachedEmbedder(dim=32)
    new_emb.model = "demo-32"
    e.agents["mina"].memory.add(e.tick, "observation", "전환 전 기억", 3)
    await rt.switch_llm(DemoBackend(), new_emb, timeout=12.0, retries=1)
    assert not e.jobs and e.router.gate.in_flight == 0 and old.closed
    assert e.triggers["mina"].get("choose_routine")  # cancelled job re-queued
    assert e.router.timeout == 12.0 and e.router.retries == 1
    assert all(r.embed_model == "demo-32" and len(r.embedding) == 32
               for r in e.agents["mina"].memory.records)
    await rt.tick_once()
    assert e.metrics["errors"] == 0


# ================================================================ review regressions
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from llm_vil.llm._http import is_llm_key_env  # noqa: E402
from llm_vil.llm.config_store import save_profile_and_key  # noqa: E402
from llm_vil.llm.embeddings import CachingEmbedder  # noqa: E402
from llm_vil.llm.factory import LLMBundle, load_dotenv  # noqa: E402
from llm_vil.server.app import embedder_reusable, host_allowed, require_local  # noqa: E402

DANGEROUS = ["PATH", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "GITHUB_TOKEN",
             "AWS_SECRET_ACCESS_KEY", "AWS_API_KEY", "STRIPE_API_KEY", "HTTPS_PROXY",
             "PYTHONPATH", "lower_api_key"]


@pytest.mark.parametrize("name", DANGEROUS)
def test_dangerous_env_names_rejected_everywhere(name, paths, client):
    assert not is_llm_key_env(name)
    with pytest.raises(ConfigError):
        save_env_key(name, SECRET, paths)
    with pytest.raises(ConfigError):
        save_profile_override("openai_compat", {"api_key_env": name}, paths=paths)
    c, _ = client
    r = c.post("/api/llm/config", json={"profile": "openai_compat",
                                        "fields": {"api_key_env": name}, "api_key": SECRET})
    assert r.status_code == 400 and SECRET not in r.text
    assert not paths.local_yaml.exists() and not paths.env_file.exists()


async def test_blocked_env_in_hand_edited_yaml_is_never_read(paths, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_never_leave")
    paths.local_yaml.write_text(
        "profiles:\n  evil:\n    provider: openai_compat\n    base_url: http://127.0.0.1:9/v1\n"
        "    fast_model: m\n    api_key_env: GITHUB_TOKEN\n", encoding="utf-8")
    s = config_summary(None, None, paths)
    evil = next(p for p in s["profiles"] if p["name"] == "evil")
    assert not evil["has_api_key"] and evil["key_hint"] is None
    res = await probe_profile("evil", paths=paths)
    assert not res["ok"] and "API_KEY" in res["error"]
    assert "ghp_should_never_leave" not in json.dumps(res)


def test_valid_llm_key_names_still_allowed():
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                 "OPENROUTER_API_KEY", "MY_VLLM_API_KEY"):
        assert is_llm_key_env(name)


def test_atomic_validation_before_any_write(paths, client):
    c, _ = client
    for body in (
        {"profile": "openrouter", "fields": {"fast_model": "x"}, "api_key": "two\nlines"},
        {"profile": "ollama_local", "fields": {"fast_model": "x"}, "api_key": SECRET},  # no env
        {"profile": "openai_compat", "fields": {"structured": "tool_call"}},
    ):
        r = c.post("/api/llm/config", json=body)
        assert r.status_code == 400, body
        assert not paths.local_yaml.exists() and not paths.env_file.exists()
    with pytest.raises(ConfigError):
        save_profile_and_key("openrouter", {"fast_model": "x"}, "bad\0key", paths=paths)
    assert not paths.local_yaml.exists()


def test_env_quote_and_backslash_roundtrip(paths, monkeypatch):
    tricky = 'sk-a"b\\c#d e\'f\\"'
    monkeypatch.setenv("TRICKY_API_KEY", "x")
    save_env_key("TRICKY_API_KEY", tricky, paths)
    monkeypatch.delenv("TRICKY_API_KEY")
    load_dotenv(paths.env_file)
    assert os.environ["TRICKY_API_KEY"] == tricky


def test_host_header_and_client_checks(client):
    c, _ = client
    for path in ("/api/llm/config", "/api/llm/stats"):
        assert c.get(path).status_code == 200
        assert c.get(path, headers={"host": "evil.example"}).status_code == 403  # rebinding
        assert c.get(path, headers={"host": "evil.example:8780"}).status_code == 403
    assert c.post("/api/llm/test", json={"profile": "demo"},
                  headers={"host": "attacker.test"}).status_code == 403
    assert host_allowed("127.0.0.1:8780", "127.0.0.1")
    assert host_allowed("localhost", "::1") and host_allowed("[::1]:8780", "::1")
    assert not host_allowed("testserver", "127.0.0.1")  # only the in-process test client
    assert not host_allowed(None, "127.0.0.1")
    from fastapi import HTTPException
    from starlette.requests import Request

    for client_addr in (None, ("10.0.0.7", 5555)):
        req = Request({"type": "http", "method": "GET", "path": "/", "headers":
                       [(b"host", b"127.0.0.1:8780")], "client": client_addr})
        with pytest.raises(HTTPException):
            require_local(req)


def test_broken_local_yaml_is_a_clean_error(paths, client, capsys):
    c, _ = client
    paths.local_yaml.write_text("profiles: [unclosed\n", encoding="utf-8")
    r = c.post("/api/llm/config", json={"profile": "demo", "fields": {}})
    assert r.status_code == 400 and "YAML" in r.json()["detail"]
    argv = ["--config", str(paths.base_yaml), "--env-file", str(paths.env_file)]
    assert cli.main([*argv, "config", "show"]) == 2
    assert cli.main([*argv, "config", "test"]) == 2
    assert "not valid YAML" in capsys.readouterr().err


class _TrackLLM(DemoBackend):
    def __init__(self, log, name):
        super().__init__()
        self.log, self.name = log, name

    async def aclose(self):
        self.log.append(f"close:{self.name}")


class _TrackEmb(CachingEmbedder):
    provider = "openai_compat"

    def __init__(self, log, name, fail_warmup=False):
        super().__init__(f"remote-{name}", 8)
        self.log, self.name, self.fail_warmup = log, name, fail_warmup

    async def warmup(self):
        if self.fail_warmup:
            raise RuntimeError("model download failed")

    def _compute(self, texts):
        return [[1.0] * 8 for _ in texts]

    def close(self):
        self.log.append(f"emb-close:{self.name}")


def test_switch_failure_closes_only_new_clients(client, monkeypatch):
    import llm_vil.server.app as app_mod

    c, app = client
    log: list[str] = []
    before = app.state.runtime.engine.router.llm
    monkeypatch.setattr(app_mod, "build_bundle", lambda *a, **k: LLMBundle(
        llm=_TrackLLM(log, "new"), embedder=_TrackEmb(log, "new", fail_warmup=True),
        profile="x", embed_profile="remote"))
    r = c.post("/api/llm/switch", json={"profile": "demo"})
    assert r.status_code == 400 and "model download failed" in r.json()["detail"]
    assert sorted(log) == ["close:new", "emb-close:new"]
    assert app.state.runtime.engine.router.llm is before  # nothing was swapped


def test_remote_embedders_are_never_reused():
    log: list[str] = []
    demo_a = LLMBundle(llm=DemoBackend(), embedder=DemoCachedEmbedder(), profile="a",
                       embed_profile="demo")
    demo_b = LLMBundle(llm=DemoBackend(), embedder=DemoCachedEmbedder(), profile="b",
                       embed_profile="demo")
    remote_a = LLMBundle(llm=DemoBackend(), embedder=_TrackEmb(log, "x"), profile="a",
                         embed_profile="openai_compat")
    remote_b = LLMBundle(llm=DemoBackend(), embedder=_TrackEmb(log, "x"), profile="b",
                         embed_profile="openai_compat")
    assert embedder_reusable(demo_a, demo_b)
    assert not embedder_reusable(remote_a, remote_b)  # base_url / key may differ
    assert not embedder_reusable(demo_a, remote_b)


def test_concurrent_switches_are_serialized_and_shutdown_closes_live_bundle(
        config, paths, monkeypatch):
    import llm_vil.server.app as app_mod

    log: list[str] = []
    counter = iter(range(100))
    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    class SlowWarmEmb(_TrackEmb):
        async def warmup(self):
            with guard:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            import asyncio as aio

            await aio.sleep(0.05)
            with guard:
                active["n"] -= 1

    def fake_build(*a, **k):
        i = next(counter)
        return LLMBundle(llm=_TrackLLM(log, f"l{i}"), embedder=SlowWarmEmb(log, f"e{i}"),
                         profile=f"p{i}", embed_profile="openai_compat")

    app = create_app(config, paused=True, llm_config_path=str(paths.base_yaml),
                     env_file=str(paths.env_file))
    with TestClient(app) as c:
        monkeypatch.setattr(app_mod, "build_bundle", fake_build)
        with ThreadPoolExecutor(4) as pool:
            codes = list(pool.map(lambda _: c.post("/api/llm/switch",
                                                   json={"profile": "demo"}).status_code,
                                  range(4)))
        assert codes == [200] * 4 and active["max"] == 1  # warmups never overlap
        st = app.state.llm_state
        assert app.state.runtime.engine.router.llm is st["bundle"].llm  # no torn state
        live = st["bundle"].llm.name
    assert f"close:{live}" in log  # shutdown closed the *current* bundle, not the first one
    closes = [x for x in log if x.startswith("close:")]
    assert len(closes) == len(set(closes)) == 4  # 3 replaced + the live one, each once


async def test_old_embedder_closed_only_after_cancelled_threads_finish(config):
    import asyncio as aio

    order: list[str] = []

    class SlowEmb(CachingEmbedder):
        provider = "local"
        blocking = True

        def __init__(self):
            super().__init__("slow-model", 8)

        def _compute(self, texts):
            time.sleep(0.2)
            order.append("compute-done")
            return [[1.0] * 8 for _ in texts]

        def close(self):
            order.append(f"close(active={self._active})")

    emb = SlowEmb()
    e = SimEngine(config, DemoBackend(), embedder=emb)
    rt = SimRuntime(e, paused=True)
    task = aio.create_task(emb.aembed(["진행 중인 임베딩"]))
    await aio.sleep(0.05)
    task.cancel()  # task returns at once, its thread keeps running
    await rt.switch_llm(DemoBackend(), DemoCachedEmbedder())
    assert order == ["compute-done", "close(active=0)"]


def test_openai_compat_rejects_tool_call(paths):
    with pytest.raises(ConfigError, match="openai_compat"):
        save_profile_override("openai_compat", {"structured": "tool_call"}, paths=paths)
    with pytest.raises(ConfigError):
        save_profile_override("anthropic", {"structured": "json_object"}, paths=paths)
    save_profile_override("openai_compat", {"structured": "prompt_only"}, paths=paths)

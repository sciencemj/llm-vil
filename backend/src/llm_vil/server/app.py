"""FastAPI app: REST (`/api/*`) + WebSocket (`/ws`). Binds 127.0.0.1 by default.

God-mode commands are local-only and unauthenticated by design; never bind publicly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import load_village
from ..engine import CognitionClock, SaveStore, SimEngine
from ..llm.config_store import (
    ConfigError,
    ConfigPaths,
    config_summary,
    known_secrets,
    load_merged_llm_config,
    probe_profile,
    save_profile_and_key,
    scrub,
)
from ..llm.factory import LLMBundle, build_bundle
from ..llm.protocol import LanguageModel
from ..schemas.config import VillageConfig
from .runtime import SimRuntime

log = logging.getLogger("llm_vil.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8780


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def origin_allowed(origin: str | None) -> bool:
    """Browsers must come from a local page; non-browser clients send no Origin."""
    if not origin:
        return True
    host = urlsplit(origin).hostname
    return host is not None and host in LOCAL_HOSTS


class ConfigBody(BaseModel):
    profile: str = Field(min_length=1, max_length=40)
    fields: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = Field(default=None, max_length=4096)  # write-only
    set_default: bool = False


class TestBody(BaseModel):
    profile: str = Field(min_length=1, max_length=40)
    fields: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = Field(default=None, max_length=4096)  # used once, never stored


class CheckpointBody(BaseModel):
    label: str = Field(default="", max_length=80)


class RestoreBody(BaseModel):
    tick: int | None = Field(default=None, ge=0)
    id: int | None = Field(default=None, ge=1)


class SwitchBody(BaseModel):
    profile: str = Field(min_length=1, max_length=40)
    embed_profile: str | None = Field(default=None, max_length=40)


LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "localhost"})
TEST_CLIENT = "testclient"  # Starlette's in-process ASGI test client


def host_allowed(host_header: str | None, client: str | None) -> bool:
    """Host must name this machine (blocks DNS-rebinding: evil.com -> 127.0.0.1)."""
    if not host_header:
        return False
    hostname = urlsplit(f"//{host_header}").hostname
    if hostname in LOCAL_HOSTS:
        return True
    return client == TEST_CLIENT and hostname == "testserver"


def require_local(request: Request) -> None:
    """Config/test/switch/stats endpoints are local-only (same rule as /ws, plus Host)."""
    client = request.client.host if request.client else None
    if client not in LOOPBACK_CLIENTS and client != TEST_CLIENT:
        raise HTTPException(status_code=403, detail="local requests only")
    if not origin_allowed(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="local requests only")
    if not host_allowed(request.headers.get("host"), client):
        raise HTTPException(status_code=403, detail="local requests only")


class KoreanJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def llm_stats_payload(runtime: SimRuntime, bundle: LLMBundle) -> dict[str, Any]:
    """Observability snapshot. Built only from public fields: never contains API keys."""
    router = runtime.engine.router
    st = router.stats
    obs = router.obs
    info = bundle.public_info()
    embed = bundle.embedder.describe()
    per_agent = {
        aid: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in a.items()}
        for aid, a in sorted(obs.per_agent.items())
    }
    return {
        "profile": info["profile"],
        "provider": info.get("provider"),
        "fast_model": info.get("fast_model"),
        "smart_model": info.get("smart_model"),
        "structured": info.get("structured", "native"),
        "base_url": info.get("base_url"),
        "embed_profile": info["embed_profile"],
        **embed,
        "embed_cache_hit_rate": round(bundle.embedder.hit_rate, 4),
        "total_calls": sum(int(a["calls"]) for a in obs.per_agent.values()),
        "attempts": st.calls,
        "valid_first_try": obs.valid_first_try,
        "repaired": st.repaired,
        "fallbacks": st.fallbacks,
        "budget_skips": obs.budget_skips,
        "timeouts": st.timeouts,
        "errors": st.errors,
        "validity_rate": round(st.validity_rate, 4),
        "tokens_in": st.tokens_in,
        "tokens_out": st.tokens_out,
        "tokens_cached": obs.tokens_cached,
        "estimated_cost_usd": round(obs.cost_usd, 6),
        "latency_ms": obs.latency_summary(),
        "in_flight": router.gate.in_flight,
        "cognition_in_flight": len(runtime.engine.jobs),
        "per_agent": per_agent,
        "recent_calls": [r.model_dump(mode="json") for r in list(obs.recent)[-40:]][::-1],
    }


def embedder_reusable(old: LLMBundle, new: LLMBundle) -> bool:
    """Reuse the warm embedder only for keyless local/demo embedders with the same profile,
    provider and model. Remote embedders (base_url / key may have changed) are rebuilt."""
    a, b = old.embedder, new.embedder
    return (
        old.embed_profile == new.embed_profile
        and a.provider == b.provider in ("local", "demo")
        and a.model == b.model and a.dim == b.dim
    )


def create_app(config: VillageConfig | None = None, llm: LanguageModel | None = None, *,
               speed: float = 1.0, paused: bool = False, store_path: str | None = None,
               autostart: bool = True, t_max_seconds: float | None = None,
               llm_profile: str | None = None, embed_profile: str | None = None,
               fresh: bool = False,
               llm_config_path: str | None = None, env_file: str | None = None) -> FastAPI:
    cfg = config or load_village()
    state: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: SaveStore | None = None
        bundle: LLMBundle | None = None
        runtime: SimRuntime | None = None
        try:
            store = SaveStore(store_path) if store_path else None
            resume: tuple[int, dict[str, Any]] | None = None
            if store is not None:
                if fresh:
                    store.reset_all()  # explicit clean start
                elif store.has_checkpoints():
                    resume = store.load_checkpoint()  # continue where the save left off
                else:
                    store.truncate_after(cfg.start_minute - 1)  # stale log, no save point
            # an explicit ``llm`` (tests) replaces the profile model: no adapter is built
            bundle = build_bundle(llm_profile, embed_profile, config_path=llm_config_path,
                                  env_file=env_file, cache=store, llm=llm)
            await bundle.embedder.warmup()  # model load runs in a worker thread
            rtcfg = bundle.runtime
            clock = CognitionClock(
                speed=speed, sim_minutes_per_real_second=cfg.clock.sim_minutes_per_real_second,
                t_max_seconds=cfg.clock.t_max_seconds if t_max_seconds is None
                else t_max_seconds,
            )
            engine = SimEngine(
                cfg, bundle.llm, store=store, clock=clock,
                checkpoint_every_days=1 if store else None, embedder=bundle.embedder,
                pricing=bundle.pricing, max_concurrency=int(rtcfg.get("max_concurrency") or 4),
                llm_timeout=bundle.llm_timeout,  # profile timeout_s wins over the runtime default
                retry_base_delay_s=float(rtcfg.get("retry_base_delay_s") or 0.5),
                json_repair_retries=bundle.repair_retries, budget=bundle.token_budget(),
            )
            if resume is not None and store is not None:
                at, data = resume
                try:
                    store.truncate_after(at)  # log written after the last save is re-run
                    engine.load_snapshot(data)
                    log.info("resumed save %s at tick %s", store.path, at)
                except Exception:  # unreadable/incompatible save: start fresh, keep file
                    log.exception("could not resume %s; starting a new village", store.path)
                    engine.load_snapshot(engine.fresh_snapshot())
                    # keep the old save points restorable, but never mix their log in
                    store.truncate_after(cfg.start_minute - 1, keep_checkpoints=True)
            runtime = SimRuntime(engine, speed=speed, paused=paused)
            state["runtime"] = runtime
            state["bundle"] = bundle
            app.state.runtime = runtime  # read-only handles for tests / tooling
            app.state.llm_state = state
            if autostart:
                runtime.start()
            yield
        finally:
            # runs on normal shutdown *and* when startup failed half-way
            if runtime is not None:
                await runtime.stop()
                if store is not None:
                    runtime.engine.flush()
            # close whatever bundle is live now (a hot-swap may have replaced the first one)
            # (the bundles it replaced were already closed by SimRuntime.switch_llm)
            live = state.get("bundle") or bundle
            if live is not None:
                await live.aclose()
            if store is not None:
                store.close()

    app = FastAPI(title="llm-vil", lifespan=lifespan, default_response_class=KoreanJSONResponse)

    def rt() -> SimRuntime:
        return state["runtime"]

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return rt().health()

    @app.get("/api/snapshot")
    async def snapshot() -> dict[str, Any]:
        return rt().snapshot()

    paths = ConfigPaths.from_args(llm_config_path, env_file)

    def bundle_now() -> LLMBundle:
        return state["bundle"]

    def summary() -> dict[str, Any]:
        b = bundle_now()
        return config_summary(b.profile, b.embed_profile, paths)

    def fail(exc: Exception, extra: list[str] | None = None) -> HTTPException:
        try:
            secrets = known_secrets(load_merged_llm_config(paths))
        except Exception:  # broken llm.local.yaml must not turn a 400 into a 500
            secrets = []
        secrets += [s for s in (extra or []) if s]
        return HTTPException(status_code=400, detail=scrub(str(exc), secrets)[:300])

    switch_lock = asyncio.Lock()  # one provider switch at a time

    @app.get("/api/llm/config")
    async def llm_config(request: Request) -> dict[str, Any]:
        require_local(request)
        return summary()

    @app.post("/api/llm/config")
    async def save_llm_config(body: ConfigBody, request: Request) -> dict[str, Any]:
        require_local(request)
        try:  # validates profile, env name and key before writing anything
            save_profile_and_key(body.profile, body.fields, body.api_key,
                                 set_default=body.set_default, paths=paths)
        except (ConfigError, OSError) as exc:
            raise fail(exc, [body.api_key or ""]) from None
        return summary()

    @app.post("/api/llm/test")
    async def test_llm(body: TestBody, request: Request) -> dict[str, Any]:
        require_local(request)
        return await probe_profile(body.profile, body.fields, api_key=body.api_key or None,
                                   paths=paths)

    @app.post("/api/llm/switch")
    async def switch_llm(body: SwitchBody, request: Request) -> dict[str, Any]:
        require_local(request)
        runtime = rt()
        async with switch_lock:  # concurrent switches cannot race warmup / aclose / state
            old = bundle_now()
            try:
                new = build_bundle(body.profile, body.embed_profile,
                                   config_path=paths.base_yaml, env_file=paths.env_file,
                                   cache=runtime.engine.store)
            except Exception as exc:
                raise fail(exc) from None
            reused = embedder_reusable(old, new)
            try:
                if reused:
                    new.embedder.close()
                    new.embedder = old.embedder  # keep warm local/demo model + caches
                else:
                    await new.embedder.warmup()  # ONNX load in a worker thread
            except Exception as exc:
                # nothing was swapped: release only what this request created
                aclose = getattr(new.llm, "aclose", None)
                if aclose is not None:
                    await aclose()
                if new.embedder is not old.embedder:
                    new.embedder.close()
                raise fail(exc) from None
            await runtime.switch_llm(new.llm, new.embedder, pricing=new.pricing,
                                     timeout=new.llm_timeout, retries=new.repair_retries)
            state["bundle"] = new
            return {"stats": llm_stats_payload(runtime, new), "config": summary()}

    # ---------------------------------------------------------------- saves
    @app.get("/api/saves/checkpoints")
    async def list_saves(request: Request) -> dict[str, Any]:
        require_local(request)
        return rt().save_info()

    @app.post("/api/saves/checkpoint")
    async def create_save(body: CheckpointBody, request: Request) -> dict[str, Any]:
        require_local(request)
        try:
            created = await rt().save_checkpoint(body.label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"created": created, **rt().save_info()}

    @app.post("/api/saves/restore")
    async def restore_save(body: RestoreBody, request: Request) -> dict[str, Any]:
        require_local(request)
        try:
            at = await rt().restore_checkpoint(tick=body.tick, checkpoint_id=body.id)
        except (ValueError, LookupError) as exc:
            raise HTTPException(status_code=400, detail=str(exc).strip("'\"")) from None
        return {"restored": at, **rt().save_info()}

    @app.post("/api/saves/reset")
    async def reset_save(request: Request) -> dict[str, Any]:
        require_local(request)
        at = await rt().reset_village()
        return {"reset": at, **rt().save_info()}

    # ---------------------------------------------------------------- replay
    @app.get("/api/replay/range")
    async def replay_range(request: Request, include_ticks: bool = False) -> dict[str, Any]:
        require_local(request)
        return rt().replay_range(include_ticks)  # full tick list only on request (~60KB)

    @app.get("/api/replay/frame")
    async def replay_frame(request: Request, tick: int) -> dict[str, Any]:
        require_local(request)
        frame = rt().replay_frame(tick)
        if frame is None:
            raise HTTPException(status_code=404, detail=f"no recorded frame at or before {tick}")
        return frame

    @app.get("/api/llm/stats")
    async def llm_stats(request: Request) -> dict[str, Any]:
        require_local(request)
        return llm_stats_payload(rt(), state["bundle"])

    @app.get("/api/agent/{actor_id}")
    async def agent(actor_id: str) -> dict[str, Any]:
        detail = rt().agent_detail(actor_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown villager {actor_id}")
        return detail

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not origin_allowed(socket.headers.get("origin")):
            await socket.close(code=1008)  # God commands are local-only
            return
        await socket.accept()
        runtime = rt()
        queue: asyncio.Queue[dict[str, Any] | None] | None = None
        pump_task: asyncio.Task[None] | None = None

        async def send(msg: dict[str, Any]) -> None:
            await socket.send_text(json.dumps(msg, ensure_ascii=False, separators=(",", ":")))

        async def pump(q: asyncio.Queue[dict[str, Any] | None]) -> None:
            while True:
                msg = await q.get()
                if msg is None:  # runtime dropped us as a slow client
                    await socket.close(code=1013)
                    return
                await send(msg)

        try:
            queue = runtime.subscribe()
            await send(runtime.snapshot())
            pump_task = asyncio.create_task(pump(queue))
            while True:
                raw = await socket.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("command must be an object")
                except ValueError as exc:
                    queue.put_nowait({"type": "ack", "ok": False, "error": str(exc)})
                    continue
                # acks share the client queue so they arrive after the deltas they caused
                queue.put_nowait(await runtime.command(msg))
        except (WebSocketDisconnect, RuntimeError):
            pass  # client left (or socket closed after a slow-client drop)
        finally:
            if queue is not None:
                runtime.unsubscribe(queue)
            if pump_task is not None:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect,
                                         RuntimeError):
                    await pump_task

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="llm-vil live village server")
    parser.add_argument("--host", default=os.environ.get("LLM_VIL_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLM_VIL_PORT",
                                                                       DEFAULT_PORT)))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--llm-profile", default=os.environ.get("LLM_VIL_PROFILE"),
                        help="profile name from config/llm.yaml (default: demo)")
    parser.add_argument("--embed-profile", default=os.environ.get("LLM_VIL_EMBED_PROFILE"),
                        help="embeddings profile: local | demo | openai_compat")
    parser.add_argument("--save", default=os.environ.get("LLM_VIL_SAVE"),
                        help="SQLite save path (enables checkpoints)")
    args = parser.parse_args()
    import uvicorn

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    app = create_app(speed=args.speed, paused=args.paused, store_path=args.save,
                     llm_profile=args.llm_profile, embed_profile=args.embed_profile)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

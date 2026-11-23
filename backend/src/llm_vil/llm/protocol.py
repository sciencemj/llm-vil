from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..schemas.base import Frozen

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- serialization helpers
def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, set | frozenset | tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Canonical JSON. ``ensure_ascii=False`` keeps Korean readable and ~2.4x cheaper."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_default,
                      separators=None if indent else (",", ":"), indent=indent)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove ``<think>`` blocks, including truncated/unclosed ones and orphan closers."""
    text = _THINK.sub("", text)
    low = text.lower()
    if "<think>" in low:  # unclosed: everything after the opener is reasoning
        text = text[: low.index("<think>")]
    elif "</think>" in low:  # opener stripped by the server: keep the answer after it
        text = text[low.rindex("</think>") + len("</think>"):]
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """Strip ``<think>`` blocks / code fences and parse the first balanced JSON object."""
    text = strip_think(text)
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("JSON root is not an object")
                return obj
    raise ValueError("unbalanced JSON object")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def stable_unit(*parts: Any) -> float:
    """Deterministic float in [0, 1) from ``parts`` (never uses Python's salted hash)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# ---------------------------------------------------------------- protocol
class LLMRequest(Frozen):
    """Prompt-cache layering: shared_prefix -> call_head -> persona_core -> user_msg."""

    purpose: str
    agent_id: str
    tick: int
    tier: Literal["fast", "deep"] = "fast"
    shared_prefix: str = ""
    call_head: str = ""
    persona_core: str = ""
    user_msg: str = ""
    response_schema: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""

    def prompt_text(self) -> str:
        parts = [self.shared_prefix, self.call_head, self.persona_core, self.user_msg]
        if self.feedback:
            parts.append(f"[이전 응답 오류] {self.feedback}")
        return "\n\n".join(p for p in parts if p)


class LLMResponse(Frozen):
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    latency_ms: float = 0.0


class ProviderError(RuntimeError):
    """Provider failure. ``retryable`` for 429/5xx; ``retry_after`` from the server header.

    Messages are built by adapters from status + (scrubbed) body only, never from headers,
    so API keys cannot leak through error text, tape entries or stats.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


class LanguageModel(Protocol):
    model: str

    async def complete_json(self, req: LLMRequest) -> LLMResponse: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Local, provider-independent, deterministic embedder (char bigrams + words).

    Stand-in for fastembed/ONNX in M1; vectors are tagged with ``model`` and ``dim``.
    """

    def __init__(self, dim: int = 64, model: str = "hashing-ko-64", cache_size: int = 4096):
        self.dim = dim
        self.model = model
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        vec = [0.0] * self.dim
        compact = re.sub(r"\s+", " ", text.lower()).strip()
        feats = compact.split() + [compact[i : i + 2] for i in range(len(compact) - 1)]
        for f in feats:
            h = hashlib.md5(f.encode("utf-8")).digest()
            vec[h[0] % self.dim] += 1.0 if h[1] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out = [round(v / norm, 6) for v in vec]
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------- runtime controls
class SharedRateGate:
    """Global concurrency cap shared by every agent's LLM calls."""

    def __init__(self, max_concurrency: int = 4) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self.in_flight = 0
        self.peak = 0

    async def __aenter__(self) -> SharedRateGate:
        await self._sem.acquire()
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.in_flight -= 1
        self._sem.release()


class TokenBudget:
    """Per-day / per-run token budget with a soft pause flag (cognition falls back)."""

    def __init__(self, per_day: int | None = None, per_run: int | None = None) -> None:
        self.per_day = per_day
        self.per_run = per_run
        self.run_tokens = 0
        self.day_tokens: dict[int, int] = {}
        self.paused = False

    def add(self, day: int, tokens: int) -> None:
        self.run_tokens += tokens
        self.day_tokens[day] = self.day_tokens.get(day, 0) + tokens
        self.paused = bool(
            (self.per_run is not None and self.run_tokens >= self.per_run)
            or (self.per_day is not None and self.day_tokens[day] >= self.per_day)
        )

    def check(self, day: int) -> bool:
        if self.per_day is not None and self.day_tokens.get(day, 0) < self.per_day:
            if self.per_run is None or self.run_tokens < self.per_run:
                self.paused = False
        return not self.paused


class TapeEntry(Frozen):
    tick: int
    agent_id: str
    purpose: str
    model: str
    request: str
    response: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    ok: bool
    attempt: int
    error: str = ""


class CallRecord(Frozen):
    tick: int
    agent_id: str
    purpose: str
    tier: str
    model: str
    status: str  # ok | repaired | fallback
    attempts: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    error: str = ""


class Pricing(Frozen):
    """USD per million tokens (0 for local/demo)."""

    input: float = 0.0
    output: float = 0.0
    cached_input: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int, tokens_cached: int) -> float:
        fresh = max(0, tokens_in - tokens_cached)
        return (fresh * self.input + tokens_cached * self.cached_input
                + tokens_out * self.output) / 1_000_000


class Observability:
    """Per-call telemetry for the LLM HUD. Never stores prompts, headers or keys."""

    def __init__(self, pricing: Pricing | dict[str, Pricing] | None = None,
                 keep_recent: int = 60, keep_latency: int = 1000) -> None:
        # one Pricing for every model, or {model_id_prefix: Pricing} (longest prefix wins)
        self.pricing: dict[str, Pricing] = (
            dict(pricing) if isinstance(pricing, dict) else {"": pricing or Pricing()})
        self.recent: deque[CallRecord] = deque(maxlen=keep_recent)
        self.latencies: deque[float] = deque(maxlen=keep_latency)
        self.per_agent: dict[str, dict[str, float]] = {}
        self.valid_first_try = 0
        self.budget_skips = 0
        self.tokens_cached = 0
        self.cost_usd = 0.0

    def record(self, rec: CallRecord) -> None:
        self.recent.append(rec)
        self.latencies.append(rec.latency_ms)
        if rec.status == "ok":
            self.valid_first_try += 1
        self.tokens_cached += rec.tokens_cached
        cost = self.price_for(rec.model).cost(rec.tokens_in, rec.tokens_out, rec.tokens_cached)
        self.cost_usd += cost
        a = self.per_agent.setdefault(rec.agent_id, {
            "calls": 0, "tokens_in": 0, "tokens_out": 0, "fallbacks": 0, "repaired": 0,
            "cost_usd": 0.0, "latency_ms_total": 0.0,
        })
        a["calls"] += 1
        a["tokens_in"] += rec.tokens_in
        a["tokens_out"] += rec.tokens_out
        a["cost_usd"] += cost
        a["latency_ms_total"] += rec.latency_ms
        if rec.status == "fallback":
            a["fallbacks"] += 1
        elif rec.status == "repaired":
            a["repaired"] += 1

    def price_for(self, model: str) -> Pricing:
        best = ""
        for prefix in self.pricing:
            if model.startswith(prefix) and len(prefix) >= len(best):
                best = prefix
        return self.pricing.get(best, Pricing())

    def latency_summary(self) -> dict[str, float]:
        xs = sorted(self.latencies)
        if not xs:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0}

        def pct(q: float) -> float:
            return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2)

        return {"p50": pct(0.5), "p95": pct(0.95), "max": round(xs[-1], 2)}


class RouterStats(BaseModel):
    calls: int = 0
    valid: int = 0
    repaired: int = 0
    fallbacks: int = 0
    timeouts: int = 0
    errors: int = 0
    budget_skips: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms_total: float = 0.0

    @property
    def validity_rate(self) -> float:
        return self.valid / self.calls if self.calls else 1.0


class LLMRouter:
    """Structured call with validator-feedback repair (<= ``retries``) and graceful fallback.

    Never raises: returns ``None`` on timeout, error, exhausted repairs or budget pause.
    """

    def __init__(self, llm: LanguageModel, *, gate: SharedRateGate | None = None,
                 budget: TokenBudget | None = None, timeout: float = 30.0,
                 retries: int = 2, ticks_per_day: int = 1440,
                 retry_base_delay_s: float = 0.5,
                 pricing: Pricing | dict[str, Pricing] | None = None,
                 max_retry_delay_s: float = 30.0) -> None:
        self.llm = llm
        self.obs = Observability(pricing)
        self.max_retry_delay_s = max_retry_delay_s
        self.gate = gate or SharedRateGate()
        self.budget = budget or TokenBudget()
        self.timeout = timeout
        self.retries = retries
        self.ticks_per_day = ticks_per_day
        self.retry_base_delay_s = retry_base_delay_s
        self.stats = RouterStats()
        self.tape: list[TapeEntry] = []

    def _tape_failure(self, req: LLMRequest, sent: LLMRequest, t0: float, attempt: int,
                      error: str) -> None:
        self.tape.append(TapeEntry(
            tick=req.tick, agent_id=req.agent_id, purpose=req.purpose,
            model=getattr(self.llm, "model", "?"),
            request=dumps({"prompt": sent.prompt_text(), "context": sent.context}), response="",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3), tokens_in=0, tokens_out=0,
            ok=False, attempt=attempt, error=error,
        ))

    def drain_tape(self) -> list[TapeEntry]:
        out, self.tape = self.tape, []
        return out

    async def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        if attempt >= self.retries:
            return
        delay = self.retry_base_delay_s * 2**attempt
        if retry_after is not None:
            delay = max(delay, retry_after)
        delay = min(delay, self.max_retry_delay_s)
        if delay > 0:
            await asyncio.sleep(delay)

    async def call(self, req: LLMRequest, model_cls: type[T],
                   validate: Callable[[T], str | None] | None = None) -> T | None:
        day = req.tick // self.ticks_per_day
        if not self.budget.check(day):
            self.stats.budget_skips += 1
            self.stats.fallbacks += 1
            self.obs.budget_skips += 1
            self.obs.record(CallRecord(
                tick=req.tick, agent_id=req.agent_id, purpose=req.purpose, tier=req.tier,
                model=getattr(self.llm, "model", "?"), status="fallback", attempts=0,
                latency_ms=0.0, tokens_in=0, tokens_out=0, tokens_cached=0,
                error="token budget exhausted (soft pause)",
            ))
            return None
        feedback = ""
        started = time.perf_counter()
        usage = {"in": 0, "out": 0, "cached": 0}
        model = getattr(self.llm, "model", "?")
        last_error = ""
        attempts = 0

        def finish(status: str) -> None:
            self.obs.record(CallRecord(
                tick=req.tick, agent_id=req.agent_id, purpose=req.purpose, tier=req.tier,
                model=model, status=status, attempts=attempts,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                tokens_in=usage["in"], tokens_out=usage["out"], tokens_cached=usage["cached"],
                error=last_error,
            ))

        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            r = req.model_copy(update={"feedback": feedback}) if feedback else req
            self.stats.calls += 1
            t0 = time.perf_counter()
            try:
                async with self.gate:
                    resp = await asyncio.wait_for(self.llm.complete_json(r), self.timeout)
            except TimeoutError:
                self.stats.timeouts += 1
                last_error = f"timeout after {self.timeout}s"
                self._tape_failure(req, r, t0, attempt, last_error)
                await self._backoff(attempt)
                continue  # transient: retry the same request
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # provider errors must never crash the sim
                self.stats.errors += 1
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                self._tape_failure(req, r, t0, attempt, last_error)
                if isinstance(exc, ProviderError) and not exc.retryable:
                    break  # e.g. 401/400: retrying cannot help
                await self._backoff(attempt, getattr(exc, "retry_after", None))
                continue  # transient: retry the same request (no validator feedback)
            latency = (time.perf_counter() - t0) * 1000
            model = resp.model
            usage["in"] += resp.tokens_in
            usage["out"] += resp.tokens_out
            usage["cached"] += resp.tokens_cached
            self.stats.tokens_in += resp.tokens_in
            self.stats.tokens_out += resp.tokens_out
            self.stats.latency_ms_total += latency
            self.budget.add(day, resp.tokens_in + resp.tokens_out)
            ok = False
            try:
                parsed = model_cls.model_validate(extract_json(resp.text))
                err = validate(parsed) if validate else None
                if err:
                    raise ValueError(err)
                ok = True
            except (ValueError, ValidationError) as exc:
                feedback = str(exc)[:300]
                last_error = f"invalid: {feedback}"[:300]
            self.tape.append(TapeEntry(
                tick=req.tick, agent_id=req.agent_id, purpose=req.purpose, model=resp.model,
                request=dumps({"prompt": r.prompt_text(), "context": r.context}),
                response=resp.text, latency_ms=round(latency, 3), tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out, ok=ok, attempt=attempt,
            ))
            if ok:
                self.stats.valid += 1
                if attempt:
                    self.stats.repaired += 1
                last_error = ""
                finish("repaired" if attempt else "ok")
                return parsed
        self.stats.fallbacks += 1
        finish("fallback")
        return None

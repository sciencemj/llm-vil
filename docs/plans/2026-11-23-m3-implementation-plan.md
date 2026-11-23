# M3 Implementation Plan: Multi-Provider LLM Layer, Local Embeddings & Observability HUD

> **Goal:** Make `llm-vil` runnable with any real LLM provider (custom OpenAI-compatible `base_url` as first-class, plus Anthropic & Gemini, alongside `DemoBackend`), with **local embeddings by default** (`fastembed`, Korean-capable multilingual ONNX model), per-vector model tagging & caching, and a live LLM Observability HUD.
> **Reference Design:** `docs/plans/2026-11-22-llm-vil-design.md` (§8, §9, §10)

## Ownership & Strict Scope
- **Coding owns:** `backend/**` and `frontend/src/**` (do NOT touch `frontend/public/assets/sprites/**` or `frontend/design-drafts/**`).
- **Security Invariant:** API keys are read strictly from environment variables / `.env` (`api_key_env` in `llm.yaml`), never stored in YAML and never exposed in REST/WebSocket/HUD/tape outputs.

## 1. Multi-Provider LLM Adapters (`backend/src/llm_vil/llm/`)
Add `httpx>=0.27` and optional/default `fastembed` dependency in `backend/pyproject.toml`.

- **`config/llm.yaml`:**
  - Profiles: `demo` (default for tests), `openai_compat` (supports custom `base_url`, `api_key_env`, `extra_headers`, `fast_model`, `smart_model`, `structured: json_schema | json_object | prompt_only`), `ollama_local` (example `http://localhost:11434/v1` preset), `anthropic` (`tool_call`), `gemini` (`json_schema`).
  - Embeddings section: `default: local` (`provider: local`, `model: intfloat/multilingual-e5-small`, `dim: 384`), plus `demo` (fast hashed BoW for tests) and optional `openai_compat` (`/embeddings`).
- **`llm/openai_compat.py`:**
  - Async `httpx.AsyncClient` calling `{base_url}/chat/completions`.
  - Respects profile `structured` mode:
    - `json_schema`: `response_format={"type": "json_schema", "json_schema": ...}` (flat schema, no `$ref`/unions)
    - `json_object`: `response_format={"type": "json_object"}` + schema instructions in system prompt
    - `prompt_only`: pure JSON prompt instruction + `<think>…</think>` stripping + outermost JSON extraction
  - Supports `extra_headers` (e.g. OpenRouter `HTTP-Referer`/`X-Title`), custom `timeout_s`, and `fast`/`smart` tier model selection.
- **`llm/anthropic_adapter.py`:**
  - Async `httpx.AsyncClient` calling Anthropic Messages API (`/v1/messages`).
  - Forced `tool_call` (`tool_choice: {"type": "tool", "name": "respond"}`) for guaranteed structured JSON + `cache_control: {"type": "ephemeral"}` on shared prefix/persona blocks.
- **`llm/gemini_adapter.py`:**
  - Async `httpx.AsyncClient` calling Gemini `generateContent` REST API.
  - Flattens Pydantic schema to Gemini's OpenAPI subset (inlines `$ref`, strips `additionalProperties`, avoids `anyOf`/unions) with `responseMimeType: "application/json"`.
- **`llm/factory.py`:**
  - Builds `(LanguageModel, Embedder)` from `llm.yaml` + profile name (`--llm-profile` / `LLM_VIL_PROFILE`) + `.env` / env vars.
  - Keeps `LLMRouter` repair loop (up to 2 validator-feedback retries), exponential backoff on 429/5xx/timeouts, token/cost/latency/validity-rate stats, and `tape` recording.

## 2. Local Embeddings by Default (`backend/src/llm_vil/llm/embeddings.py` + `engine/save_store.py`)
- **`LocalFastEmbedder`:**
  - Uses `fastembed.TextEmbedding(model_name=...)` (default `intfloat/multilingual-e5-small`, 384-d — strong Korean/English retrieval, zero API cost, works with OpenAI-compatible endpoints that have no `/embeddings`).
  - Lazy-loads model on first call inside `asyncio.to_thread(...)` so the async simulation loop never stalls.
  - Graceful error message if `fastembed` is not installed; `DemoEmbedder` remains default for `demo` profile & fast unit tests.
- **`EmbedCache` & Model Portability (`save_store.py`):**
  - SQLite `embed_cache(cache_key TEXT PRIMARY KEY, embed_model TEXT, dim INT, vector BLOB/JSON)` keyed by `sha256(embed_model + "\0" + text)`.
  - Enforce per-vector `(embed_model, dim)` tagging; `MemoryStream.retrieve` never mixes mismatched `embed_model` vectors in cosine similarity.
  - Provide `reembed_missing(store, embedder)` backfill helper when switching embedding models.

## 3. Observability API & Frontend HUD (`backend/src/llm_vil/server/` + `frontend/src/hud/`)
- **Backend:**
  - Wire `--llm-profile` & `--embed-profile` CLI args in `server/app.py` (default `demo` so existing tests stay zero-cost).
  - Add `GET /api/llm/stats` returning `{profile, provider, fast_model, smart_model, embed_provider, embed_model, embed_cache_hits, embed_cache_misses, total_calls, valid_first_try, repaired, fallbacks, validity_rate, tokens_in, tokens_out, tokens_cached, estimated_cost_usd, latency_ms: {p50, p95, max}, in_flight, per_agent: {...}, recent_calls: [...]}` (strictly no API keys!).
  - Emit lightweight `llm_stats` delta over `/ws` when calls complete (or poll cleanly on HUD open).
- **Frontend (`frontend/src/hud/llm_panel.ts`):**
  - Add top-bar toggle button `📊 LLM` opening a compact Observability drawer (Korean UI):
    - Active profile, provider, fast/smart model, local/demo embedder & cache hit rate
    - Token/cost summary + JSON validity rate + latency (`p50 / p95 / max`) + current in-flight cognition count
    - Per-agent token/call breakdown
    - Live recent LLM call log (tick, actor, call kind, tier, latency ms, tokens, status badge `OK / Repaired / Fallback`).

## 4. Verification & Acceptance Criteria (M3)
1. `uv run ruff check` clean; `uv run pytest` passes all existing 83+ tests plus new M3 tests:
   - `test_llm_adapters.py`: `httpx.MockTransport` tests for `openai_compat` (`json_schema`, `json_object`, `prompt_only`, `<think>` stripping, 429 exponential backoff, repair retry, custom `base_url` + headers), `anthropic` (forced `tool_call` + `cache_control`), `gemini` (flattened schema with no `$ref`/`anyOf`), and zero API-key leakage in stats/tape/errors.
   - `test_embeddings.py`: `EmbedCache` hit/miss keying by `(model, text)`, mixed-model cosine guard, `reembed_missing` backfill, and `LocalFastEmbedder` integration check (real local Korean embedding cosine sanity test if `fastembed` available).
   - `test_server.py`: `/api/llm/stats` endpoint & zero secret leakage.
2. `cd frontend && npm run build` passes with 0 errors.
3. Live Orca browser verification on running server: open `📊 LLM` drawer, confirm live stats/log render cleanly, **0 console errors**.

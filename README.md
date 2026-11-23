# llm-vil · 은빛골

A small autonomous village you watch rather than play. Five villagers (민아, 준호, 서연, 도윤, 하늘)
farm, chop wood, forage, cook, trade at the merchant's stall, gossip and remember — driven by an
LLM (or a free, deterministic demo brain) — rendered as crisp **sprite-stacked fake-3D pixel art**
you can rotate 360°. Inspired by Stardew Valley's daily life and Smallville's generative agents.

- **Living economy** – crops grow on a schedule, trees regrow, berry bushes refill, the notice
  board posts orders and a travelling merchant visits every 3 days.
- **Long-horizon minds** – memories with recency/importance/relevance retrieval, daily and
  importance-triggered reflections, weekly life goals, directed 4-axis relationships
  (familiarity · affection · trust · respect) and rumors that spread with fading confidence.
- **Observation + God mode** – change the weather, spawn resources, trigger village events,
  whisper rumors, then watch how they ripple.
- **Any LLM** – OpenAI-compatible endpoints (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, LiteLLM),
  Anthropic and Gemini, plus the offline `demo` profile. Korean-capable local embeddings
  (`intfloat/multilingual-e5-small`) are on by default for real providers.

## Quickstart

Requirements: Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
# backend (FastAPI + WebSocket on 127.0.0.1:8780)
cd backend
uv sync
uv run llm-vil serve            # demo brain, saves to backend/saves/village.sqlite

# frontend (Vite on 127.0.0.1:5173)
cd frontend
npm install
npm run dev
```

Open the URL Vite prints. Restarting `llm-vil serve` resumes the latest checkpoint
(`--fresh` starts a new village and wipes that save; `--memory-only` runs without a save file).
Restoring a checkpoint rewinds linearly: later checkpoints and their log/archive are removed.
`--speed 4`, `--paused`, `--llm-profile NAME`, `--embed-profile local|demo` are also available.

## Providers & API keys

Two equivalent ways:

- **In the app:** `📊 LLM → 설정` – pick or create a profile, set provider / base URL / models /
  structured mode / timeout / embedder, enter the API key (write-only), `연결 테스트`, then
  `저장 & 즉시 적용` to hot-swap the running village.
- **CLI:**
  ```bash
  uv run llm-vil config show
  uv run llm-vil config set --profile my_vllm --provider openai_compat \
      --base-url http://127.0.0.1:8000/v1 --fast-model qwen2.5-7b-instruct \
      --api-key-env MY_VLLM_API_KEY --prompt-key --embed-profile local --set-default
  uv run llm-vil config test --profile my_vllm
  uv run llm-vil serve --llm-profile my_vllm
  ```

Where things live:

| File | Contents | Git |
| --- | --- | --- |
| `backend/config/llm.yaml` | shipped profiles (demo, openai_compat, openrouter, ollama_local, anthropic, gemini) | committed |
| `backend/config/llm.local.yaml` | your non-secret overrides, merged on top | ignored |
| `backend/.env` | API keys only (`*_API_KEY`, file mode 600) | ignored |
| `backend/saves/*.sqlite` | checkpoints, event log, memory archive, LLM tape | ignored |

Keys never leave the backend: the UI and `/api/llm/*` only report whether a key is set and a
masked hint (`****1234`). All config, test, switch and save endpoints accept local requests only.

## Controls

| Input | Action |
| --- | --- |
| drag · `W` `A` `S` `D` | pan |
| right-drag · `Q` / `E` | rotate camera (`R` resets to 60°) |
| mouse wheel | zoom 2×–8× |
| click a villager | inspector: needs, plan, memories, reflections, life goals, rumors, relationship graph |
| `Space` | pause / resume · `⏭ 10분` steps while paused · `0.5x`–`8x` speed |
| `✨ 신 모드` | weather, spawn wood / berry bush / seed chest, village event, whisper a rumor |
| `🎞 리플레이` | scrub back through recorded history (`⏪ -10분` `◀ -1분` slider `▶ +1분` `⏩ +10분`, replay playback 1x–8x); the live village keeps running, `🔴 라이브 복귀` jumps back |
| `💾 저장` | save named checkpoints, restore (rewind) any of them, or reset to a fresh village |
| `📊 LLM` | live token / latency / validity stats and provider settings |
| `Esc` | close drawers |

## Architecture

```
backend/src/llm_vil/
  schemas/   frozen Pydantic models shared by everything (the only world<->agent contract)
  world/     tile grid, A*, data-driven affordances, economy; World.apply is the single validity gate
  agent/     memory stream, reflections, relationships, rumor handling, routines, conversations
  llm/       provider adapters, router (repair, backoff, budget, telemetry), embeddings, config store
  engine/    async SimEngine, cognition-aware clock, single-writer SQLite SaveStore
  server/    FastAPI REST + WebSocket delta stream, saves, provider hot-swap
  cli.py     `llm-vil` command
frontend/src/
  renderer/  Canvas 2D sprite stacking, yaw-aware depth sort, day/night tint, weather
  state/     WebSocket client + snapshot/delta store with path tweening
  hud/       top bar, inspector, timeline, God mode, LLM panel, saves panel,
             replay bar (replay_bar.ts: scrub / step / play recorded frames)
```

`world/` and `agent/` never import each other, and only `engine/save_store.py` opens SQLite —
both enforced by tests.

## Testing

```bash
cd backend
uv run ruff check
uv run pytest            # unit, adapter (mock HTTP), server, security, snapshot/restore,
                         # async race and a 30-sim-day headless soak test
cd ../frontend
npm run build            # TypeScript typecheck + production build
```

The soak test runs the demo village for 30 sim days (~10 s) and checks that nobody starves or gets
stuck, no resource goes extinct, gossip spreads and coins circulate; a separate test checks that
checkpoint restores are bit-identical.

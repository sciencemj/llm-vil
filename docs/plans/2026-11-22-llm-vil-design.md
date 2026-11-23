# llm-vil Design Document (Approach A — Revised)

> **Status:** Revised after Research Review — Pending User Approval  
> **Date:** 2026-11-22  
> **Spinoff from:** `ConflictDynamics` + `Smallville` (Generative Agents)  
> **Target:** Sprite-stacking 3D (dot 3D) Sims / Animal Crossing / Stardew Valley-style long-horizon autonomous village observation game.

---

## 1. Vision, Goals & Non-Goals

### Goals (Phase 1)
1. **Living Self-Sufficient Village (Stardew-style):** 4–6 autonomous villagers live continuously in a compact valley (cottages, farm plots, forest/trees, pond/well, village plaza/trading stall). They chop wood, till/plant/water/harvest crops, forage/cook/eat food, store/trade/gift items, rest/sleep, and socialize.
2. **Rich Agent Agency & Expressive Movement:** Agents choose routines and targets from their known world, control optional facing cues and expressions, and perform durative/interruptible actions over data-driven world objects.
3. **Sprite-Stacking 3D Visuals:** Crisp, rotatable fake-3D pixel art rendered on a dedicated HTML5 Canvas 2D renderer with 360° camera rotation, depth sorting, integer pixel scaling, and day/night lighting.
4. **Long-Horizon Memory & Evolving Relationships:** Three-tier memory (episodic hot window + daily/importance-triggered reflection + weekly core-memory synthesis with subject-indexed archive retrieval) and directed 4-axis relationships (`familiarity`, `affection`, `trust`, `respect`) with rumor/hearsay propagation.
5. **Observation + God-Mode Intervention:** Players watch smoothly (pan/zoom/rotate camera, pause/step/0.5x–8x speed, inspect mind/plan/memory/inventory/relationship graph/LLM observability HUD) and optionally intervene (weather, spawn resources, inject village events/rumors, nudge goals).
6. **Provider-Agnostic LLM Runtime:** First-class OpenAI-compatible custom endpoints (`base_url` + model for OpenAI, Ollama, vLLM, LM Studio, OpenRouter, LiteLLM), Anthropic, Gemini, and a deterministic zero-cost `DemoBackend` (with optional injected latency for async race testing).

### Non-Goals (Phase 1)
- Batch experiment matrices, Hydra sweeps, or CRAFT/ConvoKit research pipelines (dropped from `ConflictDynamics`).
- Combat, dungeons, RPG stats, or multiplayer networking.
- Playable player-controlled villager (Phase 2+ goal; Phase 1 preserves a clean `Actor` boundary so a human-controlled villager can submit atomic affordance intents directly to `World.apply` later).

---

## 2. Approaches Considered

| Approach | Summary | Tradeoffs | Verdict |
| :--- | :--- | :--- | :--- |
| **A. Python Async Sim Server + Dedicated Canvas 2D Sprite-Stacking Client** | Python/FastAPI/WebSocket/Pydantic/SQLite backend + Vite/TS pure Canvas 2D sprite-stacking renderer + DOM HUD | Small custom camera/tile math, but cleanest sprite-stacking depth sort & pixel crispness; directly reuses `ConflictDynamics` Python patterns | **Selected** |
| **B. Python Sim Server + Phaser 3 Client** | Same backend + `ConflictDynamics/viz` Phaser 3.90 viewer | Familiar scaffold, but Phaser camera rotation + pixel snap + multi-slice depth sorting conflicts with sprite-stacking | Rejected |
| **C. Full-Stack TypeScript (Node/Bun + Canvas 2D)** | Single-language backend + frontend | Shared types, but requires rewriting Python Pydantic/memory/embedding/LLM patterns from scratch | Rejected |

---

## 3. System Architecture & Reused `ConflictDynamics` Patterns

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Backend (Python 3.11+, uv, FastAPI + WebSocket, Pydantic v2, SQLite)    │
│                                                                          │
│  ┌──────────────┐  versioned View     ┌──────────────────────────────┐   │
│  │  World / Env │ ──────────────────▶ │  Agent Cognition (Async)     │   │
│  │  - TileGrid  │                     │  - Hierarchical Planner      │   │
│  │  - Affordance│ ◀─ Routine/Intent ─ │  - MemoryStream + Archive    │   │
│  │  - Economy   │   (World.apply)     │  - Directed 4-Axis Relations │   │
│  │  - A* Path   │                     │  - ConversationSession       │   │
│  │  - CognClock │                     └──────────────┬───────────────┘   │
│  └──────┬───────┘                                    │                   │
│         │ Event Deltas + Pending Writes              ▼                   │
│         │                             ┌──────────────────────────────┐   │
│         │                             │  LLM Router & Adapters       │   │
│         │                             │  - Capability-flagged JSON   │   │
│         │                             │  - SharedRateGate + Tape     │   │
│         │                             │  - Local Embeddings (384-d)  │   │
│         ▼                             └──────────────────────────────┘   │
│  Single-Writer SQLite SaveStore (WAL, schema_version, checkpoints, tape) │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │ WebSocket (change-driven deltas + God commands)
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Frontend (Vite + TypeScript + Pure Canvas 2D + DOM HUD)                 │
│  - Sprite-Stacking 3D Renderer (yaw 0–360°, pitch, integer scale)        │
│  - Client-side path tweening, depth-sorted slices, weather & day tint    │
│  - HUD: Inspector, Relationship Graph, Timeline, God-Mode, Token/Latency │
└──────────────────────────────────────────────────────────────────────────┘
```

### Layering Invariant & Reused `ConflictDynamics` Patterns
- **Strict Import Boundary:** `world/` and `agent/` never import each other; shared frozen Pydantic schemas live in `schemas/`. `World.apply()` is the single validity gate; only `agent/` mutates agent internal state (sessions return `Outcome` facts).
- **Explicitly Reused Patterns:**
  - `DemoBackend` (rule-based, deterministic, zero-cost, plus configurable simulated latency for async tests)
  - Prompt-cache layering (`shared_prefix → call_head → persona_core → per_tick_user_msg`)
  - Enum-constrained replies mapped to IDs (`to_action`) to eliminate invalid-ID refusals
  - `recent_actions` deduplication + rejection-reason feedback on retry
  - `world_notes` (adapted from `task_notes`: where my seeds are, which plot is mine, who owes me)
  - Unanswered-message tracking
  - `ensure_ascii=False` on all JSON serialization (critical for Korean prompts — avoids ~2.4× token bloat)
  - `tape.sqlite` LLM call recording for debugging/inspection
  - `SharedRateGate` concurrency cap + per-day/per-run token budget with soft pause event.

---

## 4. Clock vs. Cognition & Intent Lifecycle

### Cognition-Aware Clock (Solves Real-Time vs. LLM Latency)
- **Target Pace:** Base rate at `1x` is `1 real sec = 2 sim min` (12 real min/sim-day; scalable `0.5x`, `1x`, `2x`, `4x`, `8x`, pause, single-step).
- **Cognition-Aware Time Dilation:** Cloud calls take 2–10s and local 7–14B models take 5–20s (often serialized in Ollama). To prevent agents from standing frozen for hours of sim-time at `1x–8x`:
  - The sim clock advances normally while the longest in-flight cognition task has waited `< T_max` (default `3.0s` real time, configurable).
  - Once any required cognition exceeds `T_max`, sim-clock advancement smoothly dilates/pauses until that cognition resolves (while client-side idle animations, camera rotation, and ongoing walks continue smoothly).
- **Sleep Fast-Forward:** When all villagers are `sleeping` (typically `21:00–06:00`), the engine automatically fast-forwards to the earliest villager's staggered wake-up time (e.g., `05:30–06:30`), running overnight crop/tree growth and staggered daily reflections cleanly without wasting 37% of viewing time.
- **Staggered Wake/Sleep:** Each persona has slightly offset wake/sleep times (`±30 sim min`) to avoid thundering-herd LLM latency spikes at dawn and midnight.

### Intent Lifecycle & Concurrency Safety
- **One In-Flight Cognition per Agent:** Each agent has at most 1 active `asyncio.Task`. New perception/event triggers while thinking are coalesced into a pending trigger queue.
- **Versioned Views & Re-Validation:** Every `View` snapshot is stamped with `world_version` (monotonic tick/mutation counter). When an intent returns, `World.apply` runs on the main event loop in deterministic actor order and re-checks all preconditions (e.g., tree still standing, item still in chest). If a salient world change invalidated the view, the intent is dropped and a fresh cognition trigger is queued with the reason.
- **Cancellation on Interruption:** Incoming conversation invitations, God-mode events, or critical need thresholds cancel any in-flight routine cognition via a cancel token.
- **Timeout & Stuck-Agent Fallback:** If an LLM call times out (configurable, e.g. 30s) or fails 2 JSON repairs, the agent falls back to continuing its current plan block or executing a safe idle/forage fallback with exponential backoff — **never crashing the simulation**. If an agent hits `N=3` consecutive rejections, the engine forces a replan with the rejection reasons attached.
- **Replay Semantics:** Replay is strictly driven by the persisted **event log** (not LLM re-simulation); bit-identical deterministic re-simulation is guaranteed under `DemoBackend`.

---

## 5. Spatial Model, Facing & Routine Executors

### Tile Grid & Facing
- **Tile Grid:** 32×24 walkable tile grid (Phase 1 valley) with A* pathfinding.
- **Client-Side Tweening (Change-Driven Broadcast):** The server broadcasts state **on change** (e.g., path waypoints + start tick + speed + target facing), not at a fixed 10 Hz full-state flood. The client smoothly interpolates sub-tile `(x, y)` and rotation.
- **Automatic-First Facing:**
  - Facing (`0–360°`, snapped to 8-way or smooth angle for sprite stacking) updates **automatically** from movement direction, target object direction, and conversation partner direction.
  - To avoid wasting LLM calls on micro-movement, `facing_cue` (e.g., `face_partner`, `look_away`, `survey_area`) and `expression` are **optional fields** on routine/conversation decisions rather than standalone LLM turns. Decorative idle strolls are driven by the engine's idle behavior, not LLM calls.

### Two-Layer Action Model (Routine Executors + Atomic Affordances)
- **Why:** If the LLM chooses every atomic `till → plant → water` step per tile, calls explode to 50+/agent/day.
- **LLM Layer (Routines):** The LLM selects a **Routine** with enum-constrained parameters from the agent's *known* objects/places:
  - `tend_farm(plot_ids | "my_plots")`
  - `chop_wood(area, target_qty)`
  - `forage(area)`
  - `cook_meal(hearth_id, recipe)`
  - `eat_food(item)`
  - `store_or_take(chest_id, op, item, qty)`
  - `visit_and_talk(target_actor, topic, optional_gift_or_trade)`
  - `post_or_fulfill_order(board_id, order_id)`
  - `sleep(bed_id)` / `rest(place_id)` / `wander(area)`
- **Engine Layer (Atomic Affordances):** Each routine deterministically sequences A* pathing, facing, and durative/interruptible atomic affordances (`move_step`, `chop`, `till`, `plant`, `water`, `harvest`, `forage`, `cook`, `eat`, `sleep`, `store`, `take`, `give_item`, `trade`). A future human-controlled `Actor` (Phase 2+) can invoke these exact atomic affordances directly.
- **Data-Driven Affordance Registry (`config/village.yaml`):** Object kinds declare their states, allowed affordances, stamina/item preconditions, durations, and state transitions. The LLM prompt only lists affordances for objects currently in the agent's known/perceived spatial map.

---

## 6. Self-Sufficient Economy, Needs & Ownership Invariants

### Needs & Failure Floors
- **Needs (0–100):** `stamina` (work cost; restored by food/rest/sleep), `hunger` (gradual rise; satisfied by berries/crops/stew), `mood` (needs + social valence + weather).
- **Non-Lethal Failure Floor:** Hunger never kills an agent; hitting high hunger reduces mood and work speed, and **wild berry bushes always regrow** (2-day cycle) as a guaranteed zero-tool food fallback.

### Balanced Faucets, Sinks & Ownership
- **Seeds & Crops:** Harvesting a mature crop always yields crop + `>=1` seed (e.g., 1–2 seeds) so farming cannot permanently brick. Rain automatically waters all planted plots.
- **Wood & Trees:** Mature trees yield wood and become stumps; stumps regrow (`stump → sapling → young → mature` over 4 sim-days), with a minimum forest reserve rule (or sapling replanting).
- **Coins Faucet & Sink:**
  - *Faucet:* Village Notice Board daily/weekly village orders (e.g., "Deliver 4 wood / 3 turnips for 15 coins") and periodic Traveling Merchant arrival (every 3 sim-days).
  - *Sink:* Buying rare seeds, tools/upgrades, or specialty recipes from the Traveling Merchant, plus villager-to-villager trading.
- **Chest Ownership & Social Friction:** Every chest has `owner_id` (`actor_id` or `"shared"`). `take` from another villager's personal chest is **permitted by physics** (not hard-blocked), but generates a witnessable event if another villager is in perception range, and is discoverable by the owner when they next inspect their chest — enabling emergent trust/conflict dynamics!
- **Soak Test Invariant:** A **30-sim-day headless `DemoBackend` soak test** must pass in CI asserting: no starvation spiral, no permanent seed/tree extinction, and no villager stuck for >2 sim-hours.

---

## 7. Cognition, Conversations, Long-Horizon Memory & Relationships

### Perception & Triggers
- **Change-Only Perception:** Instead of logging every tick, the perception layer emits memories only on **state changes** (villager entered radius, object state changed, utterance heard, chest missing items, God event) with per-subject cooldowns.
- **Per-Agent Spatial Memory:** Each agent maintains a `known_world` map of discovered places/objects and their last-seen states (Smallville environment-tree pattern). Prompt enums are built from `known_world`, not omniscient world state.
- **Importance Sourcing:** No separate LLM call per memory. Engine observations use a static importance table (e.g., routine harvest = 2, witnessed theft = 8, God event = 9); conversation/decision memories piggyback `importance` + `valence` inside the existing decision JSON.

### `ConversationSession` State Machine
- **Named Targets Only:** Inviting a conversation targets specific named villager(s) (preventing `ConflictDynamics`'s co-present crowd lockup). Busy/in-conversation targets reject the invite cleanly.
- **State Machine:** `invite → accept | decline → active_turns (max 6 turns, 1 LLM call per turn) → end{silence, max_turns, interrupt, departure}`.
- **Participant Locking:** While in `active_turns`, participants pause their current durative routine (marked `in_conversation`), face each other, and step turns asynchronously one at a time. Bystanders within earshot still observe snippets as change-only perceptions.
- **Merged Turn Schema:** One fast-tier JSON call returns `{utterance, end_conversation, emotion, valence, importance, facing_cue?, gift_or_trade_offer?}`.

### Three-Tier Long-Horizon Memory & Two-Stage Retrieval
1. **Episodic Hot Window:** Recent salient memories (last 3 sim-days + all high-importance records). Retrieval uses **min-max normalized** `recency + importance + cosine_relevance + subject_boost`.
2. **Reflection (Dual Trigger):** Triggered both when cumulative importance passes threshold (e.g. `≥ 120`, Smallville-style) AND during staggered end-of-day consolidation (summarizing the day into 2–3 reflections with evidence IDs + updating `semantic_facts`). Low-importance routine observations are summarized before being archived.
3. **Weekly Core Synthesis & Two-Stage Archive Retrieval:**
   - Every 7 sim-days, daily reflections synthesize into up to **3 mutable Life Goals** (citing evidence IDs) while the **immutable `AgentSpec` persona core** is never overwritten.
   - **No Unreachable Memories:** Reflections, semantic facts, and core goals stay permanently indexed. Archived episodic memories remain searchable on demand via **SQLite subject index + FTS5** (e.g., talking to Leo automatically pulls archived Leo-related records).
4. **Hearsay:** Stored as `kind="hearsay"` with `source_actor_id` and `confidence` decaying per hop (`×0.7`), distinct from first-hand `observation`.

### Directed 4-Axis Relationships
- **Directed Graph (`a → b ≠ b → a`):** `familiarity (0..1)`, `affection (-1..+1)`, `trust (-1..+1)`, `respect (-1..+1)`, plus `summary` and `recent_key_events`.
- **Bounded Update Math:**
  - Deterministic event-delta table (e.g., gift `+0.12 affection`, fair trade `+0.10 trust`, witnessed theft `-0.35 trust`, pleasant chat `+0.05 familiarity`).
  - Optional post-conversation LLM `appraise` clamped to `±0.15` per axis to prevent drift/saturation.
  - Gentle daily decay toward neutral baseline (`0.0` for affection/trust/respect; very slow decay for familiarity).
  - Relationship `summary` refreshes on reflection or significant axis delta (`≥ 0.20`).

---

## 8. Multi-Provider LLM Layer, Embeddings & Cost Budget

### Provider Capability Matrix
Configured in `backend/config/llm.yaml` (model profiles & tiers); **API keys live strictly in env / `.env`**, never in YAML or sent to the frontend.

| Provider | `structured` Mode | Notes |
| :--- | :--- | :--- |
| `openai_compat` (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, LiteLLM) | `json_schema` / `json_object` / `prompt_only` (per-profile flag) | Custom `base_url` + optional headers; strips `<think>…</think>` reasoning blocks before JSON extraction |
| `anthropic` | `tool_call` (forced tool use) | Prompt caching on shared prefix + persona |
| `gemini` | `json_schema` (flat OpenAPI subset — no unions/`$ref`) | Flat per-call schemas with small enums |
| `demo` | Native deterministic Python | Zero-cost; supports `--latency-ms` for async race tests |

- **Schema Rule:** All LLM output schemas are kept **flat, with small enums and zero unions/$ref** so every provider parses them cleanly. Repair loop: up to 2 validator-feedback retries → graceful fallback action + per-model validity rate metric logged.

### Embedding Portability
- **Default: Local Embeddings (`fastembed` / ONNX ~384-d):** Free, fast, provider-independent, and works seamlessly with custom OpenAI-compatible endpoints that lack `/embeddings`.
- **Per-Vector Model Tagging:** Every stored vector records `(embed_model, dim)` and caches by `sha256(embed_model + text)`. Vectors from different models are never mixed in cosine similarity; switching embedding models triggers an automatic background backfill job.

### Estimated Daily Cost (5 Villagers with Routines)
- **~32 calls / ~80k tokens per villager per sim-day** → **~160 calls / ~400k tokens per sim-day** for 5 villagers (prompt-cache layering significantly reduces billed input tokens).
- At `1x` (12 real min/day = ~5 sim-days/viewing hour), mini-class models cost pennies/hour; local Ollama models work comfortably thanks to the cognition-aware clock + sleep fast-forward.

---

## 9. Persistence, Security, God-Mode & HUD

### Single-Writer SQLite Persistence
- **Single-Writer Rule:** Only `engine/save_store.py` opens SQLite (WAL mode + `busy_timeout`). `agent/` and `world/` never touch the DB; they return `pending_writes` committed in batches by the engine writer.
- **Versioning & Checkpoints:** `schema_version` table with forward migrations; `PROMPT_VERSION` tagged on memory records. Saves consist of periodic full `snapshot()/restore()` checkpoints (world + agents + clock + RNG + active sessions) plus the append-only event log after the latest checkpoint.
- **Security:** Backend binds to `127.0.0.1` by default; God commands are local-only/unauthenticated by design; `.env` secrets are never exposed over REST/WebSocket.

### God-Mode & Frontend HUD
- **God-Mode:** Weather (`Sunny`, `Rain`, `Overcast`), Spawn Resource on tile, Village Event / Targeted Rumor injection (invalidates in-flight intents, logged as `kind="external_event"` with source attribution or `hearsay` for rumors).
- **Prompt/UI Language:** Primary UI and villager dialogue/prompts in **Korean** (configurable in `village.yaml`; all JSON serialized with `ensure_ascii=False`).
- **High-Speed Readability:** At `≥4x` speed, speech bubbles auto-condense to short icons/action tags so the viewport stays readable while full dialogue remains in the timeline log.
- **Observability Panel:** Per-agent token/cost counter, live LLM call log viewer, cognition latency indicator, and model JSON validity rate.
- **Sprite-Stacking Visual Plan (Design Worker Scope):** Chunky Retro / Low Hi-Bit tier (16–24 slices villagers/props, 24–32 slices trees/cottages, `3x–6x` integer zoom) with clear front faces/hats so `facing` reads immediately at every 360° angle.

---

## 10. Milestones & Testing Strategy

| Milestone | Scope | Acceptance Criteria |
| :--- | :--- | :--- |
| **M1: Headless Demo Village** | `schemas/`, `world/`, `agent/`, `engine/`, `DemoBackend`, SQLite single-writer save/restore | **30-sim-day headless soak test** passes (0 starvation spirals, 0 resource extinction, 0 stuck agents >2h); bit-identical snapshot/restore test; async race test with injected `DemoBackend` latency |
| **M2: Sprite-Stacking Renderer + Live/Replay** | FastAPI/WebSocket server + Vite/TS Canvas 2D sprite-stacking renderer + Design worker Phase 1 asset pack + HUD | Smooth 360° camera rotation, depth sorting, crisp pixel art, speech/thought bubbles, inspector & relationship graph; **0 console errors** verified in Orca browser |
| **M3: Multi-Provider Live LLM Run** | `openai_compat`, `anthropic`, `gemini`, capability flags, `<think>` stripping, local embeddings, observability HUD | Full 1+ sim-day live run with real LLM; cognition-aware clock + sleep fast-forward verified; token/latency HUD accurate |
| **M4: God-Mode & Polish** | Weather/spawn/event/rumor controls, rumor propagation, weekly synthesis, save management | God events visibly interrupt & enter memory/hearsay; relationship graph evolves across multiple days; full Reviewer pass clean |

---

## 11. Project Layout

```text
llm-vil/
├── backend/
│   ├── pyproject.toml
│   ├── config/
│   │   ├── llm.yaml          # Provider profiles & capability flags (no API keys)
│   │   └── village.yaml      # Map, 4–6 villagers, object affordances, recipes, crops
│   ├── src/llm_vil/
│   │   ├── schemas/          # Frozen Pydantic I/O & WebSocket delta protocol
│   │   ├── world/            # TileGrid, A*, AffordanceRegistry, Economy, World.apply
│   │   ├── agent/            # Planner, Routines, MemoryStream, Archive, Relationships, Conversation
│   │   ├── llm/              # LanguageModel protocol, adapters, JSON repair, local embeddings, tape
│   │   ├── engine/           # Async SimEngine, CognitionClock, GodMode, single-writer SaveStore
│   │   └── server/           # FastAPI + WebSocket server (localhost)
│   └── tests/                # Unit, 30-day demo soak, snapshot/restore, async latency race tests
├── frontend/
│   ├── package.json
│   └── src/
│       ├── renderer/         # Pure Canvas 2D sprite-stacking engine, camera, depth sort, lighting
│       ├── state/            # WebSocket client + change-driven world reducer + tweening
│       ├── hud/              # Inspector, Relationship Graph, Timeline, God-Mode, Observability
│       └── assets/           # Sprite-stacking slice strips + manifest.json (owned by Design)
└── docs/
    └── plans/
        └── 2026-11-22-llm-vil-design.md
```

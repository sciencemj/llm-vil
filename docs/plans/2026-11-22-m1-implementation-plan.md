# M1 Implementation Plan: Headless Demo Village (`backend/`)

> **Goal:** Build the complete headless Python simulation engine for `llm-vil` under `backend/`, verified by unit tests, bit-identical snapshot/restore tests, async latency race tests, and a **30-sim-day headless `DemoBackend` soak test**.
> **Reference Design:** `docs/plans/2026-11-22-llm-vil-design.md`

## Ownership & Boundaries
- **Owned by Coding:** `backend/**` only. Do **not** touch `frontend/**` (Design owns `frontend/design-drafts/` in parallel).
- **Strict Import Boundary:** `llm_vil/world/` and `llm_vil/agent/` must **never** import each other. Shared frozen Pydantic v2 models live in `llm_vil/schemas/`. Only `llm_vil/engine/save_store.py` opens SQLite.

## Module Breakdown

### 1. Project Scaffold & Config (`backend/`)
- `backend/pyproject.toml` (Python >=3.11, `uv`, `pydantic>=2`, `pyyaml`, `pytest`, `pytest-asyncio`, `ruff`).
- `backend/config/village.yaml`:
  - 32×24 valley map layout: 5 villager cottages (with beds & personal chests), shared village chest, 10 farm plots, north pine/oak forest (12+ trees), 4 wild berry bushes, pond/well, kitchen hearth, village plaza with notice board & trading stall.
  - 5 distinct villagers (`민아`, `준호`, `서연`, `도윤`, `하늘`) with immutable persona core, staggered wake/sleep times (`05:30–06:30` / `21:00–22:00`), starting inventories, and home cottages/plots.
  - Data-driven object affordance definitions (`tree`, `farm_plot`, `berry_bush`, `hearth`, `bed`, `chest`, `notice_board`, `trading_stall`), crop growth rules (`turnip`: 2 days, `wheat`: 3 days, harvest always yields `>=1` seed), recipes (`berry_stew`, `turnip_soup`).
- `backend/config/llm.yaml`: Provider profile templates (`demo`, `openai_compat`, `anthropic`, `gemini`) with `structured` capability flags and no API keys.

### 2. Frozen Schemas (`src/llm_vil/schemas/`)
- `world.py`: `TilePos`, `Facing`, `Weather`, `WorldObjectState`, `Inventory`, `NeedsState`, `ActorSnapshot`, `WorldView` (with `world_version`), `ActionIntent`, `AtomicAction`, `ActionResult`, `WorldEvent`.
- `agent.py`: `AgentSpec` (immutable), `PlanBlock`, `RoutineChoice`, `MemoryRecord` (with `kind`, `importance`, `valence`, `subjects`, `source_actor_id`, `confidence`, `prompt_version`), `Relationship` (directed 4-axis: `familiarity`, `affection`, `trust`, `respect`, `summary`, `recent_key_events`), `ConversationTurn`, `ConversationOutcome`.
- `protocol.py`: Change-driven delta messages (`PathMoveDelta`, `ObjectChangeDelta`, `ConversationDelta`, `ClockDelta`, `GodEventDelta`) ready for M2 WebSocket streaming.

### 3. World & Economy (`src/llm_vil/world/`)
- `grid.py`: 32×24 tile grid + deterministic A* pathfinder + perception radius/line-of-sight helper.
- `affordances.py`: Data-driven registry loaded from `village.yaml` (preconditions, stamina cost, duration ticks, item yields, state transitions).
- `world.py`: `World` class with:
  - Single validity gate `apply(actor_id, atomic_action) -> ActionResult`
  - Durative action progress & interruption handling
  - Daily growth tick (crops watered or rain-watered advance; stumps regrow `stump→sapling→young→mature` over 4 days; berry bushes regrow over 2 days; notice board daily orders & traveling merchant every 3 days as coin faucet/sink)
  - Chest ownership: `take` from another's personal chest is allowed by physics and emits a witnessable/discoverable `ChestTakenEvent`
  - Needs tick (hunger rises slowly, non-lethal floor: never kills, reduces mood/speed; stamina restored by sleep/rest/food)
  - `snapshot()` / `restore()` including RNG state.

### 4. Agent Cognition, Routines, Memory & Relationships (`src/llm_vil/agent/`)
- `memory.py`: `MemoryStream` with hot window (last 3 sim-days + salient), min-max normalized retrieval (`recency + importance + relevance + subject_boost`), change-only perception dedup with per-subject cooldown, dual-trigger reflection (importance threshold ≥120 + staggered end-of-day consolidation), weekly goal synthesis (max 3 mutable goals, immutable persona core), hearsay confidence decay (`×0.7`), and pending-write queue for single-writer SQLite + two-stage subject/FTS5 archive retrieval.
- `relationships.py`: Directed 4-axis relationship store, deterministic event-delta table, clamped appraisal (`±0.15`), daily baseline decay.
- `routines.py`: Deterministic routine executors (`tend_farm`, `chop_wood`, `forage`, `cook_meal`, `eat_food`, `store_or_take`, `visit_and_talk`, `post_or_fulfill_order`, `sleep`, `rest`, `wander`) that expand an LLM routine choice into A* pathing, automatic facing, and atomic affordance steps.
- `conversation.py`: `ConversationSession` state machine (`invite → accept|decline → active_turns(max 6) → end`), named-target locking, merged 1-call-per-turn decision, returns `ConversationOutcome` facts.
- `agent.py`: `Agent` coordinator (`plan_day`, `perceive_changes`, `choose_routine`, `step_conversation`, `reflect`, `snapshot/restore`, `known_world` spatial memory, `recent_actions` dedup, `world_notes`).

### 5. LLM Protocol & `DemoBackend` (`src/llm_vil/llm/`)
- `protocol.py`: `LanguageModel` async protocol (`complete_json`, `embed`), `SharedRateGate`, token budget tracker, `ensure_ascii=False` serializer, `<think>` stripping & JSON extraction helper.
- `demo.py`: Deterministic rule-based `DemoBackend` implementing realistic Stardew villager behavior (farming, chopping, foraging, cooking, trading/orders, socializing, sleeping, reflecting) + configurable `simulated_latency_ms` for async race testing.

### 6. Engine, Cognition-Aware Clock & Single-Writer SQLite (`src/llm_vil/engine/`)
- `clock.py`: `CognitionClock` (1x = 2 sim-min/real-sec, `T_max` time dilation while cognition is in-flight, automatic sleep fast-forward when all villagers sleep).
- `save_store.py`: Single-writer SQLite store (WAL + `busy_timeout`, `schema_version` migrations, checkpoints table, append-only event log, FTS5 memory archive indexed by subject, `tape` table).
- `sim_engine.py`: `SimEngine` orchestrating the async loop: one in-flight cognition task per agent, coalesced trigger queue, `world_version` stale-view re-validation, cancel tokens on interruption, timeout/stuck-agent fallback, batched single-writer DB flush, and God-mode command hooks (`set_weather`, `spawn_resource`, `inject_event`, `inject_rumor`).

## Verification & Acceptance Criteria (M1)
1. `uv run ruff check` and `uv run pytest` pass with 0 errors.
2. **Import boundary test:** asserts `llm_vil.world` never imports `llm_vil.agent` and vice versa, and `sqlite3` is only imported in `save_store.py`.
3. **Bit-identical snapshot/restore test:** run 3 sim-days on `DemoBackend`, checkpoint at day 2, restore, run to day 3, assert identical world/agent/memory/relationship state.
4. **Async latency race test:** run with `DemoBackend(simulated_latency_ms=40)` asserting no race exceptions, stale intents cleanly dropped/re-queued, and cognition-aware clock dilates properly.
5. **30-sim-day headless soak test:** run 5 villagers for 30 full sim-days on `DemoBackend`, asserting:
   - 0 unhandled exceptions
   - 0 villagers stuck > 2 sim-hours
   - 0 permanent resource extinctions (trees > 0, berry bushes > 0, seeds/crops active)
   - Active economy (wood chopped, crops harvested, meals cooked/eaten, conversations held, relationships evolved, daily/weekly reflections & archive retrievals executed).

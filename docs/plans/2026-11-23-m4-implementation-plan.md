# M4 Implementation Plan: Save/Checkpoint Management, Rumor Propagation Polish & README

> **Goal:** Complete Phase 1 polish: persistent Save/Checkpoint management UI & API (save, list, restore/rewind, reset), organic rumor propagation & Inspector rumor/relationship polish, gentle merchant coin sink, and a clean project `README.md`.
> **Reference Design:** `docs/plans/2026-11-22-llm-vil-design.md` (§6, §7, §9, §10)

## Ownership & Strict Scope
- **Coding owns:** `backend/**`, `frontend/src/**`, `frontend/index.html`, and root `README.md`.
- **Do NOT touch:** `frontend/public/assets/sprites/**` or `frontend/design-drafts/**`.
- **Do NOT commit or push.**

## 1. Save / Checkpoint Management (`backend/src/llm_vil/engine/save_store.py`, `server/app.py`, `server/runtime.py`, `cli.py`)
- Default `llm-vil serve` to a persistent gitignored SQLite path (`backend/saves/village.sqlite`, auto-creating `saves/` and adding `saves/` to `backend/.gitignore` & root `.gitignore`), with `--memory-only` opt-out for ephemeral runs.
- Extend `SaveStore` checkpoints metadata with optional `label` (schema migration v6, backward-compatible) and `list_checkpoints(limit=50)` returning `{tick, day, minute, label, created_at}`.
- Add localhost-guarded (`require_local`) endpoints:
  - `GET /api/saves/checkpoints` → `{save_path, persistent, current_tick, current_day, checkpoints: [...]}`.
  - `POST /api/saves/checkpoint` (`{label?: string}`) → under `runtime._lock`, quiesces in-flight cognition, writes checkpoint, returns created metadata.
  - `POST /api/saves/restore` (`{tick: int}`) → under `runtime._lock`, cancels/awaits in-flight cognition, restores snapshot at `tick`, truncates/rewinds future state cleanly, and broadcasts a fresh full `snapshot` message to all connected `/ws` clients.
  - `POST /api/saves/reset` → under `runtime._lock`, resets to a fresh Day 1 village cleanly and broadcasts a fresh full `snapshot` to `/ws` clients.

## 2. Organic Rumor Propagation, Economy Sink & Inspector Polish (`backend/src/llm_vil/llm/demo.py`, `agent/`, `frontend/src/hud/inspector.ts`)
- **Organic Rumor Propagation:** When a villager has an active `hearsay` memory (from God-mode `inject_rumor` or another villager) with confidence `>= 0.35`, allow `DemoBackend` conversation turns to naturally share it with a partner who hasn't heard it yet, recording a new `hearsay` record on the listener with decayed confidence (`×0.7`) and `source_actor_id = speaker`.
- **Merchant Coin Sink:** When `merchant_present`, allow villagers with surplus coins (e.g. `>= 30`) to buy specialty/rare goods or stew/seeds from the stall so coins circulate rather than inflating monotonically over 30 days (verify 30-day soak test still passes with 0 starving/stuck/extinctions).
- **Inspector Polish (`frontend/src/hud/inspector.ts`):**
  - Display **소문 (Rumors / Hearsay)** section showing heard rumors, source villager (or `신의 속삭임`), and confidence %.
  - Display **삶의 목표 (Life Goals)** cleanly with evidence count/summary.

## 3. Save/Checkpoint Drawer UI (`frontend/src/hud/saves_panel.ts` + `main.ts` + `style.css`)
- Upgrade top-bar `💾 저장` button to toggle a compact **세이브 관리** drawer (positioned cleanly below `#topbar`, closed on `Escape` alongside Inspector/LLM/God drawers):
  - Shows current save path / persistent status & current Day/Time
  - Quick named checkpoint input + **`체크포인트 저장`** button
  - Scrollable list of saved checkpoints (`N일차 HH:MM · label`) with **`불러오기`** button (calls `POST /api/saves/restore`, updates canvas + HUD immediately)
  - **`새 마을 초기화`** button with confirm step (`POST /api/saves/reset`).

## 4. Project Documentation (`README.md`)
- Write a crisp, practical root `README.md` covering:
  - What `llm-vil` is (Stardew/Smallville-inspired sprite-stacking 3D autonomous village observation game)
  - Quickstart (`uv sync`, `npm install`, `uv run llm-vil serve`, `npm run dev`)
  - Provider & API Key setup (UI `📊 LLM → 설정` + CLI `llm-vil config show/set/test`, local Korean `e5-small` embeddings by default)
  - Controls (camera `WASD`/`Q`/`E`/scroll, click villager, God-Mode, Save/Restore)
  - Architecture & testing (`ruff`, `pytest`, 30-day soak test).

## 5. Verification & Acceptance Criteria (M4)
1. `uv run ruff check` clean; `uv run pytest` passes all 150+ tests plus new `backend/tests/test_m4_saves_rumors.py` (checkpoint list/create/restore/reset + WS snapshot broadcast + `require_local`, organic rumor propagation with `×0.7` decay, 30-day soak coin moderation & `hearsay_shared > 0`).
2. `cd frontend && npm run build` clean (0 errors).
3. Live Orca browser verification on `http://127.0.0.1:5174`:
   - Open `💾 저장` drawer → save named checkpoint → advance time → restore checkpoint → confirm Day/Time rewinds cleanly.
   - Inject God rumor → inspect target villager → verify rumor appears in Inspector `소문` section.
   - **0 console errors.**

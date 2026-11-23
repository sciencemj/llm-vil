# M2 Implementation Plan: Live Server + Sprite-Stacking 3D Canvas Renderer + HUD

> **Goal:** Connect the M1 headless village engine to a real-time FastAPI/WebSocket server and a Vite + TypeScript pure Canvas 2D sprite-stacking 3D web client with 360° camera rotation, depth sorting, HUD inspector, relationship graph, and God-mode controls.
> **Reference Design:** `docs/plans/2026-11-22-llm-vil-design.md`

## Ownership & Strict Disjoint Scope
- **Coding owns:** `backend/src/llm_vil/server/**`, `backend/pyproject.toml`, `backend/tests/test_server.py`, and `frontend/` (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/**`).
- **Do NOT touch:** `frontend/public/assets/sprites/**` or `frontend/design-drafts/**` (Design worker is actively generating the Phase 1 square-voxel sprite strips and `manifest.json` there in parallel).
- **Asset Loading Contract:** Load `/assets/sprites/manifest.json` at startup. If a sprite key is missing (or while assets are still loading), render a clean colored voxel fallback cube so the app works immediately even before every PNG lands, and automatically use the real slice strips as soon as they exist in `frontend/public/assets/sprites/`.

## 1. Backend Live Server (`backend/src/llm_vil/server/`)
- Add `fastapi`, `uvicorn[standard]`, `websockets` to `backend/pyproject.toml`.
- `backend/src/llm_vil/server/app.py`:
  - Bind default `127.0.0.1:8780` (configurable via CLI/env).
  - `GET /api/health` → `{ok, day, minute, speed, paused, actors_count}`.
  - `GET /api/snapshot` → full initial world + agents + relationships + recent events + metrics snapshot (JSON with `ensure_ascii=False`).
  - `GET /api/agent/{actor_id}` → detailed agent inspector payload (plan, routine, needs, inventory, recent memories, reflections, life goals, relationships, world_notes).
  - `WS /ws`:
    - Sends initial `snapshot` message on connect.
    - Streams change-driven deltas (`PathMoveDelta`, `ObjectChangeDelta`, `ConversationDelta`, `ClockDelta`, `GodEventDelta`, `AgentStateDelta`) as the async `SimEngine` ticks.
    - Accepts client control commands: `pause`, `resume`, `set_speed` (`0.5, 1, 2, 4, 8`), `step`, `checkpoint`, `god_weather`, `god_spawn`, `god_event`, `god_rumor`.
- Add `backend/tests/test_server.py` verifying REST endpoints, WebSocket snapshot + delta streaming, and pause/speed/God-mode commands.

## 2. Frontend Vite + TypeScript App (`frontend/`)
- Vite + TypeScript (clean vanilla DOM/TS or minimal UI — no Phaser, pure HTML5 Canvas 2D for crisp sprite stacking).
- **Renderer (`frontend/src/renderer/`):**
  - Default camera settings locked to user preference (`manifest.json` `rendererDefaults`):
    - `angleDeg = 60` (freely rotatable 0–360° via `Q`/`E` or right-drag)
    - `pitch = 0.55` (vertical squash)
    - `spacing = 0.8` (slice lift)
    - `zoom = 6` (nearest-neighbor integer scaling, wheel zoom `2x–8x`)
  - **Sprite-Stacking Engine:** Loads horizontal slice strips from `/assets/sprites/<file>`, renders bottom-to-top with symmetric `(w-1)*0.5` rotation sampling, 1px post-stack outline, and **camera-yaw-aware depth sorting** (`x*sin(yaw) + y*cos(yaw)`).
  - **World Ground & Environment:** Crisp 32×24 square-tile valley ground (grass, dirt paths, farm area, pond water, plaza stone) with subtle tile grid readability, day/night color tint overlay (`Dawn/Morning/Noon/Afternoon/Evening/Night`), and weather particles (`Rain`, `Overcast`).
  - **Client-Side Tweening:** Smoothly interpolates villager `(x, y)` along streamed path waypoints and updates `facing_deg` smoothly.
  - **Speech & Thought Bubbles:** Floating Korean speech/thought bubbles above villagers; auto-condenses to compact icons/tags at `>=4x` speed.
- **State (`frontend/src/state/`):**
  - WebSocket client with auto-reconnect + world state reducer (`snapshot` + deltas).
- **HUD (`frontend/src/hud/`):**
  - **Top Bar:** Sim Day & Clock, Weather badge, Pause/Play/Step, Speed (`0.5x–8x`), Camera Angle/Reset (`60°`), Save Checkpoint button.
  - **Right Inspector Drawer (click any villager):** Persona, current routine/plan, needs bars (`stamina`, `hunger`, `mood`), inventory chips, memories & daily reflections, life goals, and **Relationship Graph / 4-Axis bars** (`familiarity`, `affection`, `trust`, `respect`).
  - **Bottom/Side Timeline Log:** Live dialogue & village event stream (filterable).
  - **God-Mode Drawer:** Weather switcher (`Sunny`/`Rain`/`Overcast`), click-to-spawn resource (`wood`/`berry_bush`/`seeds`), inject village event, whisper rumor to a chosen villager.

## 3. Verification & Acceptance Criteria (M2)
1. `cd backend && uv run ruff check && uv run pytest` passes (including `test_server.py`).
2. `cd frontend && npm run build` (TypeScript typecheck + Vite build) passes with 0 errors.
3. **Live UI Verification (Orca Browser):** Start backend (`uvicorn`) + frontend (`vite`), open in Orca's built-in browser (`orca tab`, `orca screenshot`, `orca snapshot`, `orca console`), verify villagers are walking/acting on the 3D sprite-stacked map, camera rotates smoothly, clicking a villager opens the inspector, God-mode weather/event works, and **0 console errors**.

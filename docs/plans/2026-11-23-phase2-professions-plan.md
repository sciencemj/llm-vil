# Phase 2 Implementation Plan: 10-Profession Supply-Chain Economy, P2P Trade & Adventurer/Guild Loop

> **Goal:** Upgrade `llm-vil` from 5 self-sufficient villagers to a **10-profession interdependent village** on a **40×30 map** (with eastern Mosswood Monster Zone), featuring exclusive profession stations, P2P shop counters, escrowed villager requests & guild bounties, tool/weapon durability & repair, lightweight monster hunting, and a full Phase 2 square-voxel sprite pack.
> **Reference Research:** Phase 2 Professions/Economy Research Report.

## Strict Disjoint Ownership
- **Coding (`backend/**`, `frontend/src/**`):** Engine, schemas, YAML config, routines, DemoBackend, tests, and Canvas 2D/HUD integration. Do **NOT** touch `frontend/public/assets/sprites/**` or `frontend/design-drafts/**`.
- **Design (`frontend/public/assets/sprites/**`, `frontend/design-drafts/**`):** Phase 2 square-voxel Style 3 sprites + `manifest.json` update + `phase2-showcase.png`. Do **NOT** touch `backend/**` or `frontend/src/**`.
- **Contract:** Coding loads all new sprite keys from `/assets/sprites/manifest.json` with clean colored voxel fallbacks until Design's PNGs land.

---

## 1. Roster: 10 Interdependent Villagers (`backend/config/village.yaml`)

| ID | Name | Profession (`profession`) | Exclusive Station / Skill | Supplies | Needs via Trade |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mina` | 민아 | `farmer` (농부) | Farm plots (`farming`) | `wheat`, `turnip` | `hoe` repair, `bread`/`stew` |
| `junho` | 준호 | `woodcutter` (나무꾼) | Forest + `kiln` (`woodcutting`) | `wood`, `charcoal` | `axe` repair, food |
| `seoyeon` | 서연 | `chef_innkeeper` (요리사·여관주인) | `tavern` + `hearth` (`cooking`) | `turnip_soup`, `berry_stew`, `roast_meat` + Inn recovery | `turnip`, `boar_meat`, `wood` |
| `doyun` | 도윤 | `merchant` (상인) | `trading_stall` (`trading`) | Caravan export/import, seeds, arbitrage | Local surplus goods |
| `haneul` | 하늘 | `herbalist` (약초꾼·연금술사) | `herb_patch` + `cauldron` (`alchemy`) | `heal_potion`, `stamina_potion` + clinic recovery | `slime_gel`, `wood` |
| `taeo` | 태오 | `baker` (빵집 사장) | `bakery_oven` (`baking`) | `bread` | `wheat`, `wood` |
| `gangcheol` | 강철 | `blacksmith` (대장장이) | `forge_anvil` (`smithing`) | `axe`, `pickaxe`, `hoe`, `sword`, `armor`, tool repair | `iron_ore`, `coal`/`charcoal`, `wood`, `boar_hide` |
| `bawi` | 바위 | `miner` (광부) | `ore_node` quarry (`mining`) | `iron_ore`, `coal`, `stone` | `pickaxe` repair, food |
| `rina` | 리나 | `adventurer` (모험가) | Mosswood hunting (`combat`) | `slime_gel`, `boar_meat`, `boar_hide`, `goblin_ore` | `sword`/`armor`/repair, `heal_potion`, food |
| `hyeonwoo` | 현우 | `guildmaster` (길드장) | `guild_hall` + `guild_counter` (`guild`) | Monster drop buy-back (treasury stipend), bounties, drop resale | Monster drops from Adventurer |

---

## 2. Map Layout (40×30) & Zones
- **Village Core (`x=0..31, y=0..29`):**
  - NW: Farm district (Mina's 10 plots + Farm Shop Counter)
  - N: Forest + Kiln (Junho's trees + Timber Shop Counter)
  - NE: Quarry (`x=24..31, y=1..9`, Bawi's 6 `ore_node` rocks + Quarry Shop Counter)
  - Center Plaza (`x=11..23, y=11..18`): Notice/Bounty Board, Merchant Trading Stall, Tavern/Inn (`seoyeon`), Bakery (`taeo`), Smithy (`gangcheol`), Guild Hall (`hyeonwoo`), Well
  - SW: Pond + Herb Grove (`haneul`'s 4 `herb_patch` + Cauldron + Apothecary Counter)
  - S/W edges: Compact cottages/beds/personal chests for all 10 villagers (staggered wake/sleep times `05:15–07:00` / `20:45–22:30` to smooth LLM call spikes)
- **Mosswood Monster Zone (`x=33..39, y=2..27`):** Separated by a fence/tree border with a `gate_wood` entrance at `(32, 14)`. Monsters (`slime` ×3, `boar` ×2, `goblin` ×1) roam strictly inside `x=33..39` and never enter the village.

---

## 3. Core Mechanics (`backend/src/llm_vil/`)

### A. Profession Gating & Generalized Crafting (`world/affordances.py`, `world/world.py`)
- Add `profession: str` and `skills: list[str]` on `AgentSpec`.
- Affordances & recipes support `requires_skill`, `station`, and `requires_tool` (soft gate: working without required tool is slower or lower yield, never a hard deadlock!).
- Generalize `cook` → `craft(recipe_id)` across `hearth`, `bakery_oven`, `forge_anvil`, `cauldron`, `kiln`.

### B. Closed-Loop P2P Economy & Trading
1. **Shop Counters (`kind: shop_counter`):** Owned by producer (`owner_id`), holds `data.stock: dict[str, int]`, `data.prices: dict[str, int]`, `data.open: bool`.
   - Affordances: `stock_shop(item, qty)`, `buy_from_shop(item, qty)` (atomic coin + stock transfer, zero LLM per purchase), `set_prices(item, price)` (world enforces `0.7–1.5×` base price band).
   - Guild Counter (`guild_counter`): `sell_drops(item, qty)` buys monster drops using `data.treasury` (refilled by daily kingdom stipend, floored so buy-back never stalls) and stocks drops for resale to Chef/Herbalist/Blacksmith.
2. **Villager-Posted Escrowed Requests & Guild Bounties:**
   - Extend `Order` with `poster_id`, `escrow`, `kind: "order" | "bounty"`. `post_request` / `post_bounty` escrows coins; `fulfill_request` transfers items and pays out.
3. **Face-to-Face Offer in Conversation Turn JSON:**
   - Optional `trade_offer` (`give_item`, `give_qty`, `want_item`, `want_qty`) & `trade_response` (`accept | decline | none`) on `TurnReply`; world executes atomic swap on `accept`.

### C. Equipment, Durability, HP & Lightweight Monster Loop
- Add `hp: float = 100.0` to `NeedsState` and `equipment: dict[str, dict]` (`weapon`, `armor`, `tool` with `{item, durability, max_durability}`) to `ActorSnapshot` / runtime actor state.
- `repair_tool` affordance at `forge_anvil` (Blacksmith repairs own or others' tools for a small coin fee; failure floor: Blacksmith keeps loaner tool, Miner can hand-mine stone slowly).
- **Monsters (`kind: monster`):** Deterministic world objects (`species: slime|boar|goblin`, `state: roaming|engaged|defeated`, `respawn_tick`). Roam every few ticks inside Mosswood (`x=33..39`).
- **Combat (`fight` durative affordance):** Seeded resolution `p_win = clamp(base + weapon + armor - monster_power, 0.15, 0.95)`. Win → yields monster drops + weapon durability `-1` + HP cost; Lose or HP `< 20` → non-lethal **`fainted`** state, drops partial unbanked loot, teleports/recovers at Tavern/Apothecary clinic, logs importance-7 memory!

### D. New Routines & Economy Metrics (`agent/routines.py`, `llm/demo.py`)
- Add executors: `craft`, `stock_shop`, `buy_from_shop`, `mine`, `gather_herb`, `hunt`, `sell_drops`, `post_bounty`, `post_request`, `repair_tool`, `use_potion`, `run_stall`.
- Track economy metrics (`p2p_trades`, `shop_sales`, `bounties_completed`, `monsters_defeated`, `faints`, `tools_repaired`, `coin_supply`, `coin_gini`).

### E. Frontend Updates (`frontend/src/`)
- Update `ground.ts` for 40×30 layout (quarry stone ground in NE, Mosswood dark-moss ground + wooden border/gate in East, expanded plaza).
- Render new stations, shop counters (with subtle open/stock indicator), and roaming monsters (`monster_slime`, `monster_boar`, `monster_goblin`) with HP/combat flash.
- Update `inspector.ts` to show **직업 배지**, **HP 바**, **장비 & 내구도**, and **운영 상점 재고/가격**.

---

## 4. Verification & Acceptance Criteria (Phase 2)
1. `uv run ruff check` clean; `uv run pytest` all existing 175+ tests plus new Phase 2 tests (`test_professions_economy.py`, `test_combat_guild.py`) pass.
2. **30-sim-day 10-villager headless soak test** passes asserting:
   - Active P2P shop trades (`> 50`), crafting across bakery/forge/cauldron/kiln/hearth, monster defeats (`> 20`), guild bounties completed, tool repairs, 0 stuck >2h, 0 starving spirals, bounded coin supply.
3. `cd frontend && npm run build` passes with 0 errors.
4. Live Orca browser verification on `http://127.0.0.1:5174`: all 10 villagers, new stations, shop counters, and Mosswood monsters render cleanly in 360° square-voxel 3D with **0 console errors**.

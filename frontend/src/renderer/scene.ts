import type { SpriteBank } from "./assets";
import { Camera, TILE } from "./camera";
import { buildGround } from "./ground";
import { StackCache } from "./stack";
import type { ActorView, WorldStore } from "../state/store";
import type { Pos, WorldObject } from "../state/types";

interface Drawable {
  depth: number;
  key: string;
  angle: number;
  x: number;
  y: number;
  actorId?: string;
  obj?: WorldObject;
}

interface ScreenActor {
  id: string;
  x: number;
  y: number;
  top: number;
}

interface ScreenObj {
  obj: WorldObject;
  x: number;
  y: number;
  top: number;
}

const ROUTINE_ICON: Record<string, string> = {
  tend_farm: "🌱",
  chop_wood: "🪓",
  forage: "🫐",
  cook_meal: "🍲",
  craft: "🔨",
  eat_food: "🍽️",
  use_potion: "🧪",
  store_or_take: "📦",
  visit_and_talk: "💬",
  post_or_fulfill_order: "📜",
  post_request: "📋",
  post_bounty: "🎯",
  trade: "🪙",
  stock_shop: "🏪",
  buy_from_shop: "🛍️",
  run_stall: "🏪",
  mine: "⛏️",
  gather_herb: "🌿",
  hunt: "⚔️",
  sell_drops: "💰",
  repair_tool: "🔧",
  sleep: "💤",
  rest: "☕",
  wander: "🚶",
};

const ROUTINE_KO: Record<string, string> = {
  tend_farm: "밭일하러 가야지",
  chop_wood: "장작을 패야겠어",
  forage: "산딸기 따러 가자",
  cook_meal: "뭘 좀 만들어 먹을까",
  craft: "작업대에서 제작해야지",
  eat_food: "배고파…",
  use_potion: "물약을 마셔야겠어",
  store_or_take: "상자 정리",
  visit_and_talk: "이야기하러 가볼까",
  post_or_fulfill_order: "게시판 주문!",
  post_request: "물품 의뢰를 올려야지",
  post_bounty: "현상금을 걸어두자",
  trade: "장터에 가볼까",
  stock_shop: "가게에 물건을 진열하자",
  buy_from_shop: "가게에서 장을 보자",
  run_stall: "가게를 지키자",
  mine: "채석장에서 광석을 캐자",
  gather_herb: "약초를 채집하러 가자",
  hunt: "이끼숲으로 사냥 가자!",
  sell_drops: "길드에 전리품을 팔자",
  repair_tool: "대장간에서 장비를 수리하자",
  sleep: "이제 자야지",
  rest: "잠깐 쉬자",
  wander: "산책이나 할까",
};

const MONSTER_LABEL: Record<string, string> = {
  slime: "슬라임",
  boar: "멧돼지",
  goblin: "고블린",
};

/** Day/night multiply tint keyed by sim minute. */
const TINTS: [number, [number, number, number]][] = [
  [0, [70, 80, 140]],
  [300, [90, 96, 150]],
  [360, [236, 184, 160]], // dawn
  [480, [255, 244, 228]], // morning
  [720, [255, 255, 255]], // noon
  [960, [255, 246, 226]], // afternoon
  [1110, [244, 170, 128]], // evening
  [1230, [110, 110, 170]],
  [1440, [70, 80, 140]], // night
];

/** Half-diagonal (tiles) of the 32px cottage sprite's footprint. */
const COTTAGE_RADIUS = 1.05;

export function phaseName(minute: number): string {
  if (minute < 300 || minute >= 1230) return "밤";
  if (minute < 420) return "새벽";
  if (minute < 660) return "아침";
  if (minute < 840) return "한낮";
  if (minute < 1050) return "오후";
  return "저녁";
}

function tintFor(minute: number): [number, number, number] {
  for (let i = 0; i < TINTS.length - 1; i++) {
    const [m0, c0] = TINTS[i];
    const [m1, c1] = TINTS[i + 1];
    if (minute >= m0 && minute <= m1) {
      const t = (minute - m0) / (m1 - m0);
      return [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * t)) as [number, number, number];
    }
  }
  return [255, 255, 255];
}

function stableUnit(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
}

export function assetKeyFor(o: WorldObject): string {
  switch (o.kind) {
    case "tree":
      if (o.state === "mature") return stableUnit(o.id) < 0.5 ? "tree_oak_mature" : "tree_pine_mature";
      return `tree_${o.state}`; // stump / sapling / young
    case "farm_plot": {
      if (o.state === "untilled") return "plot_bare";
      if (o.state === "tilled") return "plot_tilled";
      if (o.state === "ready") return o.data.crop === "wheat" ? "plot_mature_wheat" : "plot_mature_turnip";
      const growth = Number(o.data.growth ?? 0);
      return growth >= 1 ? "plot_growing" : "plot_planted";
    }
    case "berry_bush":
      return o.state === "ripe" ? "berry_bush_full" : "berry_bush_empty";
    case "ore_node":
      return o.state === "rich" ? "ore_node_full" : "ore_node_empty";
    case "herb_patch":
      return o.state === "ripe" ? "herb_patch_full" : "herb_patch_empty";
    case "shop_counter":
    case "guild_counter":
      return "shop_counter";
    case "monster":
      return `monster_${String(o.data.species ?? "slime")}`;
    default:
      // hearth, chest, bed, notice_board, trading_stall, well, bakery_oven, forge_anvil, kiln, cauldron, gate_wood, guild_hall, tavern
      return o.kind;
  }
}

function lerpAngle(a: number, b: number, t: number): number {
  const d = ((((b - a) % 360) + 540) % 360) - 180;
  return a + d * t;
}

export class SceneRenderer {
  readonly camera = new Camera();
  private readonly cache = new StackCache();
  private readonly low = document.createElement("canvas");
  private ground: HTMLCanvasElement | null = null;
  private groundKey = "";
  private screenActors: ScreenActor[] = [];
  private rain: { x: number; y: number; v: number }[] = [];
  private lastFrame = performance.now();
  selectedId: string | null = null;
  followSelected = false;
  hoverTile: Pos | null = null;
  spawnPreview = false;
  dpr = 1;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly store: WorldStore,
    private readonly bank: SpriteBank,
  ) {
    bank.onChange = () => this.cache.clear();
  }

  /** Effective device-pixel scale of one base pixel (integer => crisp nearest-neighbor). */
  get pixelScale(): number {
    return Math.max(1, Math.round(this.camera.zoom * this.dpr));
  }

  resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
  }

  centerOn(x: number, y: number): void {
    this.camera.camX = x + 0.5;
    this.camera.camY = y + 0.5;
  }

  /** CSS-pixel event position -> world tile. */
  tileAt(clientX: number, clientY: number): Pos {
    const r = this.canvas.getBoundingClientRect();
    const s = this.pixelScale;
    const sx = ((clientX - r.left) * this.dpr - this.canvas.width / 2) / s;
    const sy = ((clientY - r.top) * this.dpr - this.canvas.height / 2) / s;
    const w = this.camera.unproject(sx, sy);
    return { x: Math.floor(w.x), y: Math.floor(w.y) };
  }

  /** CSS-pixel event position -> villager id under the cursor (nearest within reach). */
  actorAt(clientX: number, clientY: number): string | null {
    const r = this.canvas.getBoundingClientRect();
    const px = (clientX - r.left) * this.dpr;
    const py = (clientY - r.top) * this.dpr;
    let best: string | null = null;
    let bestD = 18 * this.pixelScale;
    for (const a of this.screenActors) {
      const cy = (a.y + a.top) / 2;
      const d = Math.hypot(px - a.x, py - cy);
      if (d < bestD) {
        bestD = d;
        best = a.id;
      }
    }
    return best;
  }

  /** CSS-pixel pan (left-drag). */
  panBy(dxCss: number, dyCss: number): void {
    const s = this.pixelScale;
    this.camera.pan((dxCss * this.dpr) / s, (dyCss * this.dpr) / s);
    this.followSelected = false;
  }

  private cottages: Pos[] = [];
  private cottageKey = "";

  /**
   * Decorative cottages are not sim objects, so they get a world-space spot behind each
   * home's bed/chest row that does not overlap any object footprint. With no overlap,
   * plain depth sorting is correct from every camera angle (no depth hacks).
   */
  private cottageSpots(): Pos[] {
    const objs = [...this.store.objects.values()].filter((o) => o.kind !== "farm_plot");
    const key = `${objs.length}|${this.store.areas.length}`;
    if (key === this.cottageKey) return this.cottages;
    this.cottageKey = key;
    const RC = COTTAGE_RADIUS;
    const radius = (kind: string) => (kind === "tree" ? 0.8 : 0.6);
    this.cottages = [];
    for (const area of this.store.areas) {
      if (!area.id.startsWith("home_")) continue;
      const [ax, ay, aw] = area.rect;
      const ideal = { x: ax + aw / 2, y: ay - 0.4 };
      let best = ideal;
      let bestScore = Infinity;
      for (let cx = ax + 1; cx <= ax + aw - 1 + 1e-6; cx += 0.25) {
        for (let cy = ay - 1.6; cy <= ay + 0.2 + 1e-6; cy += 0.1) {
          let overlap = 0;
          for (const o of objs) {
            const d = Math.hypot(o.pos.x + 0.5 - cx, o.pos.y + 0.5 - cy);
            overlap += Math.max(0, RC + radius(o.kind) - d);
          }
          for (const c of this.cottages) overlap += Math.max(0, 2 * RC - Math.hypot(c.x - cx, c.y - cy));
          const score = overlap * 100 + Math.hypot(cx - ideal.x, cy - ideal.y);
          if (score < bestScore) {
            bestScore = score;
            best = { x: cx, y: cy };
          }
        }
      }
      this.cottages.push(best);
    }
    return this.cottages;
  }

  private updateTweens(now: number, dt: number): void {
    const tick = this.store.predictedTick(now);
    const smooth = 1 - Math.exp(-dt * 14);
    for (const a of this.store.actors.values()) {
      let tx = a.x;
      let ty = a.y;
      let facing = a.facing;
      if (a.path && a.pathStart) {
        const pts = [a.pathStart, ...a.path.waypoints];
        const f = (tick - a.path.start_tick) / a.path.step_ticks;
        if (f >= pts.length - 1) {
          a.path = null;
          a.pathStart = null;
        } else {
          const fi = Math.max(0, f);
          const i = Math.floor(fi);
          const t = fi - i;
          const p0 = pts[i];
          const p1 = pts[Math.min(i + 1, pts.length - 1)];
          tx = p0.x + (p1.x - p0.x) * t;
          ty = p0.y + (p1.y - p0.y) * t;
          if (p1.x !== p0.x || p1.y !== p0.y) facing = (Math.atan2(p1.y - p0.y, p1.x - p0.x) * 180) / Math.PI;
        }
      }
      if (Math.hypot(tx - a.dispX, ty - a.dispY) > 4) {
        a.dispX = tx; // teleport-size jumps (fast-forward, reconnect) snap
        a.dispY = ty;
      } else {
        a.dispX += (tx - a.dispX) * smooth;
        a.dispY += (ty - a.dispY) * smooth;
      }
      a.dispFacing = lerpAngle(a.dispFacing, facing, Math.min(1, smooth * 1.2));
    }
  }

  frame(now: number): void {
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    const store = this.store;
    const cam = this.camera;
    const s = this.pixelScale;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#1d2a24";
    ctx.fillRect(0, 0, W, H);
    if (!store.ready) return;

    this.updateTweens(now, dt);
    if (this.followSelected && this.selectedId) {
      const a = store.actors.get(this.selectedId);
      if (a) {
        cam.camX += (a.dispX + 0.5 - cam.camX) * Math.min(1, dt * 4);
        cam.camY += (a.dispY + 0.5 - cam.camY) * Math.min(1, dt * 4);
      }
    }

    const lw = Math.ceil(W / s);
    const lh = Math.ceil(H / s);
    if (this.low.width !== lw || this.low.height !== lh) {
      this.low.width = lw;
      this.low.height = lh;
    }
    const lctx = this.low.getContext("2d");
    if (!lctx) return;
    lctx.imageSmoothingEnabled = false;
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.fillStyle = "#2c4636";
    lctx.fillRect(0, 0, lw, lh);
    const cx = Math.round(lw / 2);
    const cy = Math.round(lh / 2);

    // ground: R(yaw) then vertical squash, identical to Camera.project
    const gk = `${store.width}x${store.height}`;
    if (!this.ground || this.groundKey !== gk) {
      this.ground = buildGround(store.width, store.height, store.water, store.areas);
      this.groundKey = gk;
    }
    lctx.setTransform(1, 0, 0, cam.pitch, cx, cy);
    lctx.rotate(cam.rad);
    lctx.translate(-cam.camX * TILE, -cam.camY * TILE);
    lctx.drawImage(this.ground, 0, 0);
    if (this.spawnPreview && this.hoverTile) {
      lctx.fillStyle = "rgba(255, 240, 120, 0.55)";
      lctx.fillRect(this.hoverTile.x * TILE, this.hoverTile.y * TILE, TILE, TILE);
    }
    if (this.selectedId) {
      const a = store.actors.get(this.selectedId);
      if (a) {
        lctx.strokeStyle = "rgba(255, 236, 140, 0.95)";
        lctx.lineWidth = 1.5;
        lctx.beginPath();
        lctx.arc((a.dispX + 0.5) * TILE, (a.dispY + 0.5) * TILE, TILE * 0.55, 0, Math.PI * 2);
        lctx.stroke();
      }
    }
    lctx.setTransform(1, 0, 0, 1, 0, 0);

    // sprites, depth-sorted by camera-yaw-aware depth (x*sin + y*cos)
    const items: Drawable[] = [];
    for (const o of store.objects.values()) {
      if (o.kind === "monster" && o.state === "defeated") continue;
      const p = cam.project(o.pos.x + 0.5, o.pos.y + 0.5);
      items.push({ depth: p.depth, key: assetKeyFor(o), angle: cam.angleDeg, x: p.sx, y: p.sy, obj: o });
    }
    for (const c of this.cottageSpots()) {
      const p = cam.project(c.x, c.y);
      items.push({ depth: p.depth, key: "cottage", angle: cam.angleDeg, x: p.sx, y: p.sy });
    }
    for (const a of store.actors.values()) {
      const p = cam.project(a.dispX + 0.5, a.dispY + 0.5);
      // sprite front faces local +y (south); rotate it to the villager's facing
      items.push({
        depth: p.depth + 0.01,
        key: `villager_${a.actor_id}`,
        angle: cam.angleDeg + a.dispFacing - 90,
        x: p.sx,
        y: p.sy,
        actorId: a.actor_id,
      });
    }
    items.sort((a, b) => a.depth - b.depth);
    const screenActors: ScreenActor[] = [];
    const screenObjs: ScreenObj[] = [];
    for (const it of items) {
      const sx = cx + it.x;
      const sy = cy + it.y;
      if (sx < -80 || sx > lw + 80 || sy < -80 || sy > lh + 120) continue;
      const img = this.cache.get(it.key, this.bank.get(it.key), it.angle, cam.pitch, cam.spacing);
      const dx = Math.round(sx - img.anchorX);
      const dy = Math.round(sy - img.anchorY);
      lctx.drawImage(img.canvas, dx, dy);
      if (it.actorId) screenActors.push({ id: it.actorId, x: sx * s, y: sy * s, top: dy * s });
      if (it.obj && (it.obj.kind === "monster" || it.obj.kind === "shop_counter" || it.obj.kind === "guild_counter")) {
        screenObjs.push({ obj: it.obj, x: sx * s, y: sy * s, top: dy * s });
      }
    }
    this.screenActors = screenActors;

    // upscale (integer, nearest-neighbor) onto the display canvas
    ctx.drawImage(this.low, 0, 0, lw * s, lh * s);

    // day/night tint + weather
    const minute = store.clock.minute;
    const [tr, tg, tb] = tintFor(minute);
    ctx.globalCompositeOperation = "multiply";
    ctx.fillStyle = `rgb(${tr},${tg},${tb})`;
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
    this.drawWeather(ctx, W, H, dt);

    // Phase 2 object overlays (shop stock indicators & Mosswood monster badges)
    for (const so of screenObjs) this.drawObjectBadge(ctx, so, now);

    // bubbles & name tags (full-res text)
    // replay shows full lines unless it is auto-playing at >= 4x
    const condensed = store.replaying ? store.replayCondensed : store.clock.speed >= 4;
    for (const sa of screenActors) {
      const a = store.actors.get(sa.id);
      if (a) this.drawLabel(ctx, a, sa, now, condensed);
    }
  }

  private drawObjectBadge(ctx: CanvasRenderingContext2D, so: ScreenObj, now: number): void {
    const dpr = this.dpr;
    const o = so.obj;
    if (o.kind === "monster") {
      const species = String(o.data.species ?? "slime");
      const engaged = o.state === "engaged";
      const label = engaged ? `⚔️ ${MONSTER_LABEL[species] ?? species}` : (MONSTER_LABEL[species] ?? species);
      const fontPx = Math.round(10 * dpr);
      ctx.font = `700 ${fontPx}px "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const bw = ctx.measureText(label).width + 8 * dpr;
      const pulse = engaged ? 0.8 + 0.2 * Math.sin(now / 120) : 0.72;
      ctx.fillStyle = engaged ? `rgba(196, 48, 42, ${pulse})` : "rgba(38, 28, 36, 0.68)";
      roundRect(ctx, so.x - bw / 2, so.top - fontPx * 1.4, bw, fontPx * 1.35, 5 * dpr);
      ctx.fill();
      ctx.fillStyle = engaged ? "#fff5d6" : "#f0d4d4";
      ctx.fillText(label, so.x, so.top - fontPx * 0.72);
      return;
    }
    const stock = (o.data.stock ?? {}) as Record<string, number>;
    const total = Object.values(stock).reduce((acc, n) => acc + (Number(n) || 0), 0);
    if (total <= 0) return;
    const isGuild = o.kind === "guild_counter";
    const fontPx = Math.round(9.5 * dpr);
    ctx.font = `700 ${fontPx}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const tag = `${isGuild ? "🛡️" : "🏪"}${total}`;
    const bw = ctx.measureText(tag).width + 7 * dpr;
    ctx.fillStyle = isGuild ? "rgba(56, 86, 138, 0.82)" : "rgba(44, 110, 66, 0.82)";
    roundRect(ctx, so.x - bw / 2, so.top - fontPx * 1.3, bw, fontPx * 1.3, 5 * dpr);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, so.x, so.top - fontPx * 0.65);
  }

  private drawWeather(ctx: CanvasRenderingContext2D, W: number, H: number, dt: number): void {
    const weather = this.store.clock.weather;
    if (weather === "overcast" || weather === "rain") {
      ctx.fillStyle = weather === "rain" ? "rgba(40, 52, 70, 0.28)" : "rgba(90, 96, 110, 0.22)";
      ctx.fillRect(0, 0, W, H);
    }
    if (weather !== "rain") {
      this.rain.length = 0;
      return;
    }
    const target = Math.round((W * H) / 9000);
    while (this.rain.length < target) {
      this.rain.push({ x: Math.random() * W, y: Math.random() * H, v: 0.8 + Math.random() * 0.6 });
    }
    ctx.strokeStyle = "rgba(190, 215, 255, 0.55)";
    ctx.lineWidth = Math.max(1, this.dpr);
    ctx.beginPath();
    const len = 14 * this.dpr;
    for (const d of this.rain) {
      d.y += d.v * 900 * dt * this.dpr;
      d.x -= d.v * 160 * dt * this.dpr;
      if (d.y > H) {
        d.y = -len;
        d.x = Math.random() * W * 1.2;
      }
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(d.x - len * 0.18, d.y + len);
    }
    ctx.stroke();
  }

  private drawLabel(ctx: CanvasRenderingContext2D, a: ActorView, sa: ScreenActor, now: number, condensed: boolean): void {
    const dpr = this.dpr;
    const fontPx = Math.round(12 * dpr);
    ctx.font = `600 ${fontPx}px "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const nameY = sa.y + 10 * dpr;
    const prefix = a.fainted ? "💫 " : a.activity === "fight" ? "⚔️ " : "";
    const label = `${prefix}${a.name}`;
    const nw = ctx.measureText(label).width + 10 * dpr;
    ctx.fillStyle = a.fainted
      ? "rgba(180, 56, 56, 0.88)"
      : a.actor_id === this.selectedId
        ? "rgba(255, 228, 120, 0.95)"
        : "rgba(20, 24, 30, 0.72)";
    roundRect(ctx, sa.x - nw / 2, nameY - fontPx * 0.75, nw, fontPx * 1.5, 6 * dpr);
    ctx.fill();
    ctx.fillStyle = a.actor_id === this.selectedId && !a.fainted ? "#2a2010" : "#fff";
    ctx.fillText(label, sa.x, nameY);

    if (typeof a.hp === "number" && a.hp < 99.5) {
      const barW = 28 * dpr;
      const barH = 3.5 * dpr;
      const bx = sa.x - barW / 2;
      const by = nameY + fontPx * 0.9;
      ctx.fillStyle = "rgba(20, 20, 20, 0.75)";
      ctx.fillRect(bx, by, barW, barH);
      ctx.fillStyle = a.hp < 35 ? "#e04848" : "#6ec85a";
      ctx.fillRect(bx, by, barW * Math.max(0, Math.min(1, a.hp / 100)), barH);
    }

    const b = a.bubble;
    if (!b) return;
    if (now > b.until) {
      a.bubble = null;
      return;
    }
    let text = b.kind === "speech" ? b.text : ROUTINE_KO[b.text] ?? b.text;
    if (condensed) text = b.kind === "speech" ? "💬" : ROUTINE_ICON[b.text] ?? "💭";
    else if (b.kind === "thought") text = `${ROUTINE_ICON[b.text] ?? "💭"} ${text}`;
    const maxW = 220 * dpr;
    if (condensed) ctx.font = `${Math.round(18 * dpr)}px "Apple Color Emoji", "Noto Color Emoji", sans-serif`;
    const lines = wrap(ctx, text, maxW);
    const lineH = fontPx * 1.3;
    const bw = Math.min(maxW, Math.max(...lines.map((l) => ctx.measureText(l).width))) + 16 * dpr;
    const bh = lines.length * lineH + 10 * dpr;
    const bx = sa.x - bw / 2;
    const by = sa.top - bh - 10 * dpr;
    const fade = Math.min(1, (b.until - now) / 400);
    ctx.globalAlpha = fade;
    ctx.fillStyle = b.kind === "speech" ? "rgba(255, 252, 240, 0.96)" : "rgba(232, 240, 255, 0.9)";
    ctx.strokeStyle = "rgba(40, 36, 30, 0.85)";
    ctx.lineWidth = 1.5 * dpr;
    roundRect(ctx, bx, by, bw, bh, 9 * dpr);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    if (b.kind === "speech") {
      ctx.moveTo(sa.x - 5 * dpr, by + bh);
      ctx.lineTo(sa.x, by + bh + 8 * dpr);
      ctx.lineTo(sa.x + 5 * dpr, by + bh);
      ctx.fill();
    } else {
      ctx.arc(sa.x, by + bh + 6 * dpr, 3 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = "#2b2620";
    lines.forEach((l, i) => ctx.fillText(l, sa.x, by + 5 * dpr + lineH * (i + 0.5)));
    ctx.globalAlpha = 1;
  }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const ch of Array.from(text)) {
    const next = line + ch;
    if (ctx.measureText(next).width > maxW && line) {
      out.push(line);
      line = ch;
      if (out.length === 2) break;
    } else {
      line = next;
    }
  }
  if (out.length < 3 && line) out.push(out.length === 2 ? `${line}…` : line);
  return out.slice(0, 3);
}

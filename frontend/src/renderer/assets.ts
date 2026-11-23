/**
 * Sprite bank: loads `/assets/sprites/manifest.json` + horizontal slice strips.
 * Any sprite that is missing (or still loading) renders as a colored voxel fallback cube,
 * and missing sprites are retried periodically so Design's strips are picked up live.
 */

export interface AssetDef {
  file: string;
  frameWidth: number;
  frameHeight: number;
  slices: number;
  footprintTiles?: [number, number];
}

export interface RendererDefaults {
  pitch: number;
  spacing: number;
  defaultAngleDeg: number;
  defaultZoom: number;
}

export interface SliceSet {
  slices: Uint8ClampedArray[];
  w: number;
  h: number;
  real: boolean;
  version: number;
}

export const LOCKED_DEFAULTS: RendererDefaults = { pitch: 0.55, spacing: 0.8, defaultAngleDeg: 60, defaultZoom: 6 };

const BASE = "/assets/sprites/";
const RETRY_MS = 10_000;

const FALLBACK_COLORS: [string, [number, number, number]][] = [
  ["villager_mina", [232, 112, 120]],
  ["villager_junho", [92, 132, 196]],
  ["villager_seoyeon", [236, 176, 76]],
  ["villager_doyun", [146, 106, 196]],
  ["villager_haneul", [96, 186, 160]],
  ["villager", [220, 190, 160]],
  ["tree_stump", [128, 92, 60]],
  ["tree", [70, 150, 80]],
  ["plot_mature", [214, 186, 96]],
  ["plot_growing", [120, 170, 80]],
  ["plot_planted", [118, 92, 64]],
  ["plot", [140, 104, 70]],
  ["berry_bush_full", [180, 60, 90]],
  ["berry_bush", [84, 140, 84]],
  ["hearth", [170, 90, 70]],
  ["chest", [176, 128, 72]],
  ["bed", [200, 200, 220]],
  ["notice_board", [150, 110, 80]],
  ["trading_stall", [210, 140, 90]],
  ["well", [130, 140, 150]],
  ["cottage", [196, 150, 110]],
];

const FALLBACK_SHAPE: Record<string, [number, number]> = {
  villager: [8, 14],
  tree: [12, 22],
  tree_stump: [10, 4],
  tree_sapling: [6, 8],
  plot: [14, 2],
  plot_growing: [12, 5],
  plot_mature: [12, 8],
  berry_bush: [12, 9],
  cottage: [28, 26],
  trading_stall: [20, 14],
  well: [12, 12],
  bed: [14, 5],
};

function lookup<T>(key: string, table: [string, T][] | Record<string, T>, dflt: T): T {
  const entries = Array.isArray(table) ? table : Object.entries(table);
  let best: T | undefined;
  let bestLen = -1;
  for (const [prefix, v] of entries) {
    if (key.startsWith(prefix) && prefix.length > bestLen) {
      best = v;
      bestLen = prefix.length;
    }
  }
  return best ?? dflt;
}

/** Voxel box made of `count` identical slices with shaded edges and a lighter cap. */
export function fallbackSlices(key: string): SliceSet {
  const [size, height] = lookup(key, FALLBACK_SHAPE, [12, 12] as [number, number]);
  const [r, g, b] = lookup(key, FALLBACK_COLORS, [180, 180, 180] as [number, number, number]);
  const w = size + 4;
  const h = size + 4;
  const count = Math.max(2, height);
  const slices: Uint8ClampedArray[] = [];
  for (let z = 0; z < count; z++) {
    const data = new Uint8ClampedArray(w * h * 4);
    const top = z === count - 1;
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const edge = x === 2 || y === 2 || x === w - 3 || y === h - 3;
        const k = top ? 1.18 : edge ? 0.78 : 1.0;
        const i = (y * w + x) * 4;
        data[i] = Math.min(255, r * k);
        data[i + 1] = Math.min(255, g * k);
        data[i + 2] = Math.min(255, b * k);
        data[i + 3] = 255;
      }
    }
    slices.push(data);
  }
  return { slices, w, h, real: false, version: 0 };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

function extractSlices(img: HTMLImageElement, def: AssetDef): Uint8ClampedArray[] {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(img, 0, 0);
  const count = Math.min(def.slices, Math.floor(img.width / def.frameWidth));
  const out: Uint8ClampedArray[] = [];
  for (let z = 0; z < count; z++) out.push(ctx.getImageData(z * def.frameWidth, 0, def.frameWidth, def.frameHeight).data);
  return out;
}

export class SpriteBank {
  defaults: RendererDefaults = { ...LOCKED_DEFAULTS };
  private defs = new Map<string, AssetDef>();
  private loaded = new Map<string, SliceSet>();
  private fallbacks = new Map<string, SliceSet>();
  private inflight = new Set<string>();
  private retryTimer = 0;
  onChange: () => void = () => {};

  get stats(): { real: number; total: number } {
    return { real: this.loaded.size, total: this.defs.size };
  }

  /** True once every sprite in the manifest has real slices. */
  get complete(): boolean {
    return this.defs.size > 0 && [...this.defs.keys()].every((k) => this.loaded.has(k));
  }

  async init(): Promise<void> {
    await this.refreshManifest();
    if (!this.complete) {
      this.retryTimer = window.setInterval(() => void this.refreshManifest(), RETRY_MS);
    }
  }

  dispose(): void {
    window.clearInterval(this.retryTimer);
    this.retryTimer = 0;
  }

  /** Stop polling as soon as the manifest is fully loaded. */
  private settle(): void {
    if (this.complete && this.retryTimer) this.dispose();
  }

  private async refreshManifest(): Promise<void> {
    try {
      const res = await fetch(`${BASE}manifest.json`, { cache: "no-store" });
      if (!res.ok) return;
      const manifest = (await res.json()) as { rendererDefaults?: Partial<RendererDefaults>; assets?: Record<string, AssetDef> };
      // Camera defaults are locked by the user; the manifest may only confirm them.
      this.defaults = { ...LOCKED_DEFAULTS };
      for (const [key, def] of Object.entries(manifest.assets ?? {})) {
        const prev = this.defs.get(key);
        this.defs.set(key, def);
        if (!this.loaded.has(key) || (prev && prev.file !== def.file)) void this.loadAsset(key, def);
      }
    } catch {
      // manifest not available yet: fallback cubes keep rendering
    }
  }

  private async loadAsset(key: string, def: AssetDef): Promise<void> {
    if (this.inflight.has(key)) return;
    this.inflight.add(key);
    try {
      const head = await fetch(`${BASE}${def.file}`, { method: "HEAD", cache: "no-store" });
      const type = head.headers.get("content-type") ?? "";
      if (!head.ok || !type.startsWith("image/")) return; // not written yet: keep fallback
      const img = await loadImage(`${BASE}${def.file}?v=${Date.now()}`);
      const slices = extractSlices(img, def);
      if (slices.length === 0) return;
      const prev = this.loaded.get(key);
      this.loaded.set(key, { slices, w: def.frameWidth, h: def.frameHeight, real: true, version: (prev?.version ?? 0) + 1 });
      this.onChange();
      this.settle();
    } catch {
      // keep fallback; retried on next manifest refresh
    } finally {
      this.inflight.delete(key);
    }
  }

  get(key: string): SliceSet {
    const real = this.loaded.get(key);
    if (real) return real;
    let fb = this.fallbacks.get(key);
    if (!fb) {
      fb = fallbackSlices(key);
      this.fallbacks.set(key, fb);
    }
    return fb;
  }
}

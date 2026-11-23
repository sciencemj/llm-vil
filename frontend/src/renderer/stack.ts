import type { SliceSet } from "./assets";

export interface StackImage {
  canvas: HTMLCanvasElement;
  /** Pixel in `canvas` that sits on the object's ground anchor (center of slice 0). */
  anchorX: number;
  anchorY: number;
}

const OUTLINE: [number, number, number] = [24, 28, 36];

/**
 * Render a sprite stack bottom-to-top with symmetric `(w-1)*0.5` rotation sampling and a
 * 1px post-stack outline. `rad` rotates sprite-local axes onto screen axes.
 */
export function renderStack(set: SliceSet, rad: number, pitch: number, spacing: number): StackImage {
  const { slices, w, h } = set;
  const count = slices.length;
  const diag = Math.ceil(Math.hypot(w, h)) + 2;
  const liftMax = Math.ceil((count - 1) * spacing);
  const bw = diag + 4;
  const bh = Math.ceil(diag * pitch) + liftMax + 4;
  const cx = bw >> 1;
  const baseY = bh - Math.ceil(diag * pitch * 0.5) - 2;
  const ca = Math.cos(rad);
  const sa = Math.sin(rad);
  const body = new Uint8ClampedArray(bw * bh * 4);
  const halfW = (w - 1) * 0.5;
  const halfH = (h - 1) * 0.5;
  for (let z = 0; z < count; z++) {
    const sl = slices[z];
    const originY = baseY - Math.round(z * spacing);
    for (let py = 0; py < bh; py++) {
      const dy = (py - originY) / pitch;
      for (let px = 0; px < bw; px++) {
        const dx = px - cx;
        const sx = Math.round(dx * ca + dy * sa + halfW);
        const sy = Math.round(-dx * sa + dy * ca + halfH);
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
        const si = (sy * w + sx) * 4;
        if (sl[si + 3] === 0) continue;
        const bi = (py * bw + px) * 4;
        body[bi] = sl[si];
        body[bi + 1] = sl[si + 1];
        body[bi + 2] = sl[si + 2];
        body[bi + 3] = 255;
      }
    }
  }
  const out = new Uint8ClampedArray(bw * bh * 4);
  for (let py = 1; py < bh - 1; py++) {
    for (let px = 1; px < bw - 1; px++) {
      const i = (py * bw + px) * 4;
      if (body[i + 3] > 0) {
        out[i] = body[i];
        out[i + 1] = body[i + 1];
        out[i + 2] = body[i + 2];
        out[i + 3] = 255;
      } else if (
        body[i - bw * 4 + 3] > 0 ||
        body[i + bw * 4 + 3] > 0 ||
        body[i - 4 + 3] > 0 ||
        body[i + 4 + 3] > 0
      ) {
        out[i] = OUTLINE[0];
        out[i + 1] = OUTLINE[1];
        out[i + 2] = OUTLINE[2];
        out[i + 3] = 255;
      }
    }
  }
  const canvas = document.createElement("canvas");
  canvas.width = bw;
  canvas.height = bh;
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.putImageData(new ImageData(out, bw, bh), 0, 0);
  return { canvas, anchorX: cx, anchorY: baseY };
}

/** LRU cache of rendered stacks keyed by asset, version and quantized angle. */
export class StackCache {
  private map = new Map<string, StackImage>();
  constructor(private readonly limit = 900, private readonly stepDeg = 3) {}

  get(key: string, set: SliceSet, angleDeg: number, pitch: number, spacing: number): StackImage {
    const bucket = (Math.round(((angleDeg % 360) + 360) % 360 / this.stepDeg) * this.stepDeg) % 360;
    const id = `${key}|${set.real ? set.version : "fb"}|${bucket}|${pitch}|${spacing}`;
    const hit = this.map.get(id);
    if (hit) {
      this.map.delete(id);
      this.map.set(id, hit);
      return hit;
    }
    const img = renderStack(set, (bucket * Math.PI) / 180, pitch, spacing);
    this.map.set(id, img);
    if (this.map.size > this.limit) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
    return img;
  }

  clear(): void {
    this.map.clear();
  }
}

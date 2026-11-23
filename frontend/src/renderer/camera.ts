import { LOCKED_DEFAULTS } from "./assets";

export const TILE = 16; // base pixels per tile (matches 16px villager frames)
export const MIN_ZOOM = 2;
export const MAX_ZOOM = 8;

/**
 * Yaw-rotatable orthographic camera. World (tile) coordinates map to base pixels:
 *   p = R(yaw) * ((x - camX) * TILE, (y - camY) * TILE);  screen = (p.x, p.y * pitch)
 * which is exactly the inverse of the sprite-stack sampling, so sprites and ground agree.
 */
export class Camera {
  angleDeg = LOCKED_DEFAULTS.defaultAngleDeg;
  pitch = LOCKED_DEFAULTS.pitch;
  spacing = LOCKED_DEFAULTS.spacing;
  zoom = LOCKED_DEFAULTS.defaultZoom;
  camX = 16;
  camY = 12;

  get rad(): number {
    return (this.angleDeg * Math.PI) / 180;
  }

  rotate(deltaDeg: number): void {
    this.angleDeg = (((this.angleDeg + deltaDeg) % 360) + 360) % 360;
  }

  reset(): void {
    this.angleDeg = LOCKED_DEFAULTS.defaultAngleDeg;
    this.pitch = LOCKED_DEFAULTS.pitch;
    this.spacing = LOCKED_DEFAULTS.spacing;
    this.zoom = LOCKED_DEFAULTS.defaultZoom;
  }

  setZoom(z: number): void {
    this.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Math.round(z)));
  }

  /** World tile coords -> base-pixel offset from the viewport center, plus depth. */
  project(x: number, y: number): { sx: number; sy: number; depth: number } {
    const c = Math.cos(this.rad);
    const s = Math.sin(this.rad);
    const dx = (x - this.camX) * TILE;
    const dy = (y - this.camY) * TILE;
    const px = dx * c - dy * s;
    const py = dx * s + dy * c;
    return { sx: px, sy: py * this.pitch, depth: py };
  }

  /** Base-pixel offset from the viewport center -> world tile coords (ground plane). */
  unproject(sx: number, sy: number): { x: number; y: number } {
    const c = Math.cos(this.rad);
    const s = Math.sin(this.rad);
    const px = sx;
    const py = sy / this.pitch;
    const dx = px * c + py * s;
    const dy = -px * s + py * c;
    return { x: dx / TILE + this.camX, y: dy / TILE + this.camY };
  }

  /** Pan by a screen-space delta measured in base pixels. */
  pan(dsx: number, dsy: number): void {
    const a = this.unproject(0, 0);
    const b = this.unproject(dsx, dsy);
    this.camX -= b.x - a.x;
    this.camY -= b.y - a.y;
  }
}

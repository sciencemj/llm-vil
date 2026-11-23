import { TILE } from "./camera";
import type { Area } from "../state/types";

type RGB = [number, number, number];

const GRASS: RGB = [118, 168, 86];
const FOREST: RGB = [86, 136, 70];
const DIRT: RGB = [178, 146, 104];
const SOIL: RGB = [126, 92, 62];
const STONE: RGB = [176, 170, 158];
const WATER: RGB = [74, 140, 196];
const HOME: RGB = [150, 176, 98];

function hash(x: number, y: number, s: number): number {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function inRect(x: number, y: number, r: number[]): boolean {
  return x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3];
}

/** Bakes the 32x24 square-tile valley ground into one texture (TILE px per tile). */
export function buildGround(width: number, height: number, water: number[][], areas: Area[]): HTMLCanvasElement {
  const area = (id: string) => areas.find((a) => a.id === id)?.rect;
  const tiles: RGB[][] = [];
  for (let y = 0; y < height; y++) {
    const row: RGB[] = [];
    for (let x = 0; x < width; x++) {
      let c: RGB = GRASS;
      const forest = area("forest");
      if (forest && inRect(x, y, forest)) c = FOREST;
      for (const a of areas) if (a.id.startsWith("home_") && inRect(x, y, a.rect)) c = HOME;
      if (y === 11 && x >= 2 && x <= 29) c = DIRT; // village lane
      if (x === 13 && y >= 11 && y <= 17) c = DIRT; // lane to plaza
      if (x === 20 && y >= 11 && y <= 12) c = DIRT;
      const farm = area("farm");
      if (farm && inRect(x, y, farm)) c = SOIL;
      const plaza = area("plaza");
      const kitchen = area("kitchen");
      if ((plaza && inRect(x, y, plaza)) || (kitchen && inRect(x, y, kitchen))) c = STONE;
      for (const r of water) if (inRect(x, y, r)) c = WATER;
      row.push(c);
    }
    tiles.push(row);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width * TILE;
  canvas.height = height * TILE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  const img = ctx.createImageData(canvas.width, canvas.height);
  const d = img.data;
  for (let ty = 0; ty < height; ty++) {
    for (let tx = 0; tx < width; tx++) {
      const base = tiles[ty][tx];
      const isWater = base === WATER;
      for (let py = 0; py < TILE; py++) {
        for (let px = 0; px < TILE; px++) {
          const n = hash(tx * TILE + px, ty * TILE + py, 7);
          let k = 0.94 + n * 0.1;
          if (!isWater && n > 0.97) k = 1.12; // sparse highlight specks
          if (isWater && (py + Math.floor(n * 3)) % 6 === 0) k = 1.1; // ripple lines
          if (px === 0 || py === 0) k *= 0.9; // subtle tile grid
          const i = ((ty * TILE + py) * canvas.width + tx * TILE + px) * 4;
          d[i] = Math.min(255, base[0] * k);
          d[i + 1] = Math.min(255, base[1] * k);
          d[i + 2] = Math.min(255, base[2] * k);
          d[i + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

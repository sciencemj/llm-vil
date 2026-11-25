import "./style.css";
import { SpriteBank } from "./renderer/assets";
import { SceneRenderer } from "./renderer/scene";
import { GodPanel } from "./hud/god";
import { Inspector } from "./hud/inspector";
import { LLMPanel } from "./hud/llm_panel";
import { SavesPanel } from "./hud/saves_panel";
import { ReplayBar } from "./hud/replay_bar";
import { Timeline } from "./hud/timeline";
import { Topbar } from "./hud/topbar";
import { toast } from "./hud/util";
import { WorldStore } from "./state/store";
import { SimSocket, type ConnState } from "./state/ws";
import type { ServerMessage } from "./state/types";

function must<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} missing`);
  return node as T;
}

const canvas = must<HTMLCanvasElement>("world");
const store = new WorldStore();
const bank = new SpriteBank();
const scene = new SceneRenderer(canvas, store, bank);
let conn: ConnState = "connecting";
let centered = false;

const socket = new SimSocket(
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  (msg: ServerMessage) => {
    if (msg.type === "ack") {
      if (!msg.ok) toast(`⚠️ ${msg.error ?? "명령 실패"}`);
      else if (msg.cmd === "checkpoint") toast("💾 체크포인트를 저장했어요");
      return;
    }
    if (store.replaying) return; // 🎞 replay owns the view; re-synced on "라이브 복귀"
    const wasLimited = store.clock.limit_reached === true;
    store.apply(msg);
    if (!wasLimited && store.clock.limit_reached) {
      toast(`⏰ 최대 실행 일수(${store.clock.max_run_days}일)에 도달해 자동 일시정지되었습니다`);
    }
    if (msg.type === "snapshot") {
      god.syncVillagers();
      if (!centered) {
        const first = store.actors.values().next().value;
        if (first) scene.centerOn(first.x, first.y);
        centered = true;
      }
    }
    if (store.needsResync) {
      store.needsResync = false;
      void resync();
    }
  },
  (s) => {
    conn = s;
  },
);

async function resync(): Promise<void> {
  try {
    const res = await fetch("/api/snapshot");
    if (res.ok) store.apply((await res.json()) as ServerMessage);
  } catch {
    // retried on the next unknown object
  }
}

const send = (cmd: Record<string, unknown>) => {
  if (!socket.send(cmd)) toast("서버에 연결되지 않았어요");
};

const inspector = new Inspector(must("inspector"), store);
inspector.onClose = () => {
  scene.selectedId = null;
  scene.followSelected = false;
};
const timeline = new Timeline(must("timeline"), store);
const god = new GodPanel(must("god"), store, {
  send,
  setPlacing: (kind) => {
    scene.spawnPreview = kind !== null;
    canvas.classList.toggle("placing", kind !== null);
  },
});
const llmPanel = new LLMPanel(must("llm"), store);
const replay = new ReplayBar(must("replay"), store);
const savesPanel = new SavesPanel(must("saves"), () => store.clock, (days) => send({ cmd: "set_max_days", days }));
savesPanel.beforeRewind = () => replay.exit(); // never hold a stale range across a rewind
// inspector, LLM and saves share the right-hand slot: opening one closes the others
llmPanel.onOpen = () => {
  inspector.close();
  if (savesPanel.open) savesPanel.toggle(false);
};
savesPanel.onOpen = () => {
  inspector.close();
  if (llmPanel.open) llmPanel.toggle(false);
};
const topbar = new Topbar(must("topbar"), store, scene.camera, {
  send,
  toggleGod: () => god.toggle(),
  toggleLLM: () => llmPanel.toggle(),
  toggleSaves: () => savesPanel.toggle(),
  toggleReplay: () => void replay.toggle(),
  rotateCamera: (deltaDeg) => scene.camera.rotate(deltaDeg),
  resetCamera: () => scene.camera.reset(),
});

function select(id: string): void {
  if (llmPanel.open) llmPanel.toggle(false);
  if (savesPanel.open) savesPanel.toggle(false);
  scene.selectedId = id;
  scene.followSelected = true;
  inspector.open(id);
}

// ---------------------------------------------------------------- input
let drag: { x: number; y: number; button: number; moved: boolean } | null = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, button: e.button, moved: false };
});
canvas.addEventListener("pointermove", (e) => {
  if (god.placing) scene.hoverTile = scene.tileAt(e.clientX, e.clientY);
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  drag.moved = true;
  canvas.classList.add("dragging");
  if (drag.button === 2) scene.camera.rotate(dx * 0.4);
  else scene.panBy(dx, dy);
  drag.x = e.clientX;
  drag.y = e.clientY;
});
canvas.addEventListener("pointerup", (e) => {
  canvas.classList.remove("dragging");
  const wasClick = drag && !drag.moved && drag.button === 0;
  drag = null;
  if (!wasClick) return;
  if (god.placing) {
    const t = scene.tileAt(e.clientX, e.clientY);
    send({ cmd: "god_spawn", kind: god.placing, x: t.x, y: t.y });
    god.setPlacing(null);
    return;
  }
  const id = scene.actorAt(e.clientX, e.clientY);
  if (id) select(id);
});
canvas.addEventListener("pointercancel", () => {
  drag = null;
  canvas.classList.remove("dragging");
});
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    scene.camera.setZoom(scene.camera.zoom + (e.deltaY < 0 ? 1 : -1));
  },
  { passive: false },
);
const heldMove = new Set<string>();
const MOVE_KEYS: Record<string, [number, number]> = {
  w: [0, 1],
  s: [0, -1],
  a: [1, 0],
  d: [-1, 0],
  arrowup: [0, 1],
  arrowdown: [0, -1],
  arrowleft: [1, 0],
  arrowright: [-1, 0],
};

function isEditableTarget(t: EventTarget | null): boolean {
  return (
    t instanceof HTMLInputElement ||
    t instanceof HTMLSelectElement ||
    t instanceof HTMLTextAreaElement ||
    (t instanceof HTMLElement && t.isContentEditable)
  );
}

window.addEventListener("keydown", (e) => {
  if (isEditableTarget(e.target)) return;
  const k = e.key.toLowerCase();
  if (!e.metaKey && !e.ctrlKey && !e.altKey && k in MOVE_KEYS) {
    if (k.startsWith("arrow")) e.preventDefault();
    heldMove.add(k);
    return;
  }
  if (e.key === "q" || e.key === "Q") scene.camera.rotate(-15);
  else if (e.key === "e" || e.key === "E") scene.camera.rotate(15);
  else if (e.key === "r" || e.key === "R") scene.camera.reset();
  else if (e.key === " ") {
    e.preventDefault();
    if (replay.active) replay.togglePlay(); // Space drives the recording while replaying
    else send({ cmd: store.clock.paused ? "resume" : "pause" });
  } else if (e.key === "Escape") {
    inspector.close();
    if (llmPanel.open) llmPanel.toggle(false);
    if (savesPanel.open) savesPanel.toggle(false);
    god.toggle(false);
  }
});
window.addEventListener("keyup", (e) => {
  heldMove.delete(e.key.toLowerCase());
});
window.addEventListener("blur", () => {
  heldMove.clear();
});
window.addEventListener("resize", () => scene.resize());

// Keep drawers below the top bar even when it wraps onto several rows.
const topbarEl = must("topbar");
const syncTopbarOffset = () => {
  const r = topbarEl.getBoundingClientRect();
  document.documentElement.style.setProperty("--topbar-bottom", `${Math.ceil(r.bottom)}px`);
};
new ResizeObserver(syncTopbarOffset).observe(topbarEl);
window.addEventListener("resize", syncTopbarOffset);
syncTopbarOffset();
window.addEventListener("pagehide", () => {
  bank.dispose();
  llmPanel.dispose();
  socket.close();
});

// ---------------------------------------------------------------- loop
const PAN_SPEED = 560; // screen px/sec
const PAN_DAMP = 14; // exponential smoothing rate
let panVx = 0;
let panVy = 0;
let lastFrame = 0;
let lastHud = 0;
function loop(now: number): void {
  const dt = lastFrame ? Math.min(0.05, (now - lastFrame) / 1000) : 0;
  lastFrame = now;
  let tx = 0;
  let ty = 0;
  for (const k of heldMove) {
    const d = MOVE_KEYS[k];
    if (d) {
      tx += d[0];
      ty += d[1];
    }
  }
  const len = Math.hypot(tx, ty);
  const targetVx = len > 0 ? (tx / len) * PAN_SPEED : 0;
  const targetVy = len > 0 ? (ty / len) * PAN_SPEED : 0;
  const alpha = 1 - Math.exp(-PAN_DAMP * dt);
  panVx += (targetVx - panVx) * alpha;
  panVy += (targetVy - panVy) * alpha;
  if (Math.hypot(panVx, panVy) > 2) {
    scene.panBy(panVx * dt, panVy * dt);
  } else if (len === 0) {
    panVx = 0;
    panVy = 0;
  }

  scene.frame(now);
  if (now - lastHud > 200) {
    lastHud = now;
    topbar.update(conn, bank.stats, replay.active);
    timeline.render();
  }
  requestAnimationFrame(loop);
}

scene.resize();
void bank.init();
socket.connect();
requestAnimationFrame(loop);

// debug handle for live inspection in the browser console
(window as unknown as { __vil: unknown }).__vil = { store, scene, bank, select };

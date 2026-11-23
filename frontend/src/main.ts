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
    store.apply(msg);
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
const savesPanel = new SavesPanel(must("saves"), () => store.clock);
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
window.addEventListener("keydown", (e) => {
  const t = e.target;
  if (
    t instanceof HTMLInputElement ||
    t instanceof HTMLSelectElement ||
    t instanceof HTMLTextAreaElement ||
    (t instanceof HTMLElement && t.isContentEditable)
  ) {
    return;
  }
  const PAN: Record<string, [number, number]> = { w: [0, 1], s: [0, -1], a: [1, 0], d: [-1, 0] };
  const pan = PAN[e.key.toLowerCase()];
  if (pan) {
    scene.panBy(pan[0] * 120, pan[1] * 120); // screen-relative, works at any camera angle
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
let lastHud = 0;
function loop(now: number): void {
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

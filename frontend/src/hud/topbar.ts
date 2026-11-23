import type { Camera } from "../renderer/camera";
import { phaseName } from "../renderer/scene";
import type { WorldStore } from "../state/store";
import type { ConnState } from "../state/ws";
import { el, fmtClock, WEATHER_KO } from "./util";

const SPEEDS = [0.5, 1, 2, 4, 8];

export interface TopbarActions {
  send: (cmd: Record<string, unknown>) => void;
  toggleGod: () => void;
  toggleLLM: () => void;
  toggleSaves: () => void;
  toggleReplay: () => void;
  rotateCamera: (deltaDeg: number) => void;
  resetCamera: () => void;
}

export class Topbar {
  private clock = el("div", { class: "clock" });
  private weather = el("span", { class: "badge" });
  private play = el("button", { class: "btn", title: "일시정지/재생 (Space)" });
  private stepBtn = el("button", { class: "btn", title: "10분 진행" }, "⏭ 10분");
  private speedBtns = new Map<number, HTMLButtonElement>();
  private angle = el("button", { class: "btn", title: "카메라 60° 회전 (Q/E · 우클릭 드래그)" });
  private conn = el("span", { class: "conn", title: "서버 연결" });
  private assets = el("span", { class: "badge", title: "로드된 스프라이트" });

  constructor(root: HTMLElement, private store: WorldStore, private camera: Camera, actions: TopbarActions) {
    this.play.addEventListener("click", () => actions.send({ cmd: store.clock.paused ? "resume" : "pause" }));
    const step = this.stepBtn;
    step.addEventListener("click", () => actions.send({ cmd: "step", ticks: 10 }));
    const speeds = el("div", { class: "group", role: "group", "aria-label": "속도" });
    for (const s of SPEEDS) {
      const b = el("button", { class: "btn" }, `${s}x`);
      b.addEventListener("click", () => actions.send({ cmd: "set_speed", speed: s }));
      this.speedBtns.set(s, b);
      speeds.append(b);
    }
    const camGroup = el("div", { class: "group", role: "group", "aria-label": "카메라" });
    this.angle.addEventListener("click", () => actions.rotateCamera(60));
    const resetCam = el("button", { class: "btn", title: "시점 초기화 (60° · 기본 줌 · R)" }, "↺");
    resetCam.addEventListener("click", actions.resetCamera);
    camGroup.append(this.angle, resetCam);
    const save = el("button", { class: "btn", title: "세이브 관리 (체크포인트·되돌리기·초기화)" }, "💾 저장");
    save.addEventListener("click", actions.toggleSaves);
    const god = el("button", { class: "btn", title: "신 모드" }, "✨ 신 모드");
    god.addEventListener("click", actions.toggleGod);
    const llm = el("button", { class: "btn", title: "LLM 관측 (토큰·지연·호출 로그)" }, "📊 LLM");
    llm.addEventListener("click", actions.toggleLLM);
    this.replayBtn.addEventListener("click", actions.toggleReplay);
    root.append(
      this.conn, this.clock, this.weather, el("span", { class: "sep" }), this.play, step, speeds,
      el("span", { class: "sep" }), camGroup, this.replayBtn, save, god, llm, this.assets,
    );
  }

  private replayBtn = el("button", { class: "btn", title: "지난 장면 되감기·재생 (라이브는 계속 진행)" }, "🎞 리플레이");

  update(conn: ConnState, assetStats: { real: number; total: number }, replaying = false): void {
    this.replayBtn.classList.toggle("on", replaying);
    const c = this.store.clock;
    this.clock.innerHTML = "";
    this.clock.append(`${c.day + 1}일차 ${fmtClock(c.minute)}`, el("small", {}, phaseName(c.minute)));
    this.weather.textContent = WEATHER_KO[c.weather] ?? c.weather;
    // in replay the clock is a recorded frame's: live controls are locked, not mislabeled
    this.play.textContent = replaying ? "🎞 리플레이 중" : c.paused ? "▶ 재생" : "⏸ 정지";
    this.play.disabled = replaying;
    this.stepBtn.disabled = replaying;
    for (const [s, b] of this.speedBtns) {
      b.classList.toggle("on", !replaying && s === c.speed);
      b.disabled = replaying;
    }
    this.angle.textContent = `↻ ${Math.round(this.camera.angleDeg)}°`;
    this.conn.className = `conn ${conn}`;
    this.assets.textContent = `🧱 ${assetStats.real}/${assetStats.total}`;
  }
}

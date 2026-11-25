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
  private limit = el("button", { class: "badge limit" });
  private thinking = el("span", { class: "badge thinking", role: "status", "aria-live": "polite" });

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
      this.limit, this.thinking,
    );
    this.limit.addEventListener("click", actions.toggleSaves);
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
    const max = c.max_run_days ?? null;
    const reached = !replaying && c.limit_reached === true;
    this.limit.hidden = replaying;
    this.limit.classList.toggle("reached", reached);
    this.limit.textContent = reached
      ? `⏰ ${max}일 한도 도달`
      : max === null ? "한도: 무제한" : `한도: ${Math.floor(c.run_days_elapsed ?? 0)}/${max}일`;
    this.limit.title = reached
      ? `▶ 재생을 누르면 ${max}일 더 진행합니다`
      : "이만큼 진행하면 자동으로 일시정지합니다 (클릭: 세이브 관리에서 변경)";
    const n = replaying ? 0 : c.thinking_count ?? 0;
    this.thinking.hidden = n <= 0;
    if (n > 0) {
      const names = (c.thinking_actors ?? []).map((id) => this.store.name_of(id)).join(", ");
      this.thinking.textContent = `💭 생각 중 (${n}명)`;
      this.thinking.title = `모두의 결정이 모이면 함께 움직입니다: ${names}`;
    }
  }
}

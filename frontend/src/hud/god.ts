import type { WorldStore } from "../state/store";
import { el } from "./util";

export interface GodActions {
  send: (cmd: Record<string, unknown>) => void;
  setPlacing: (kind: string | null) => void;
}

const SPAWNS: [string, string][] = [
  ["wood", "🌳 나무(장작)"],
  ["berry_bush", "🫐 산딸기 덤불"],
  ["seeds", "🌱 씨앗 상자"],
];

export class GodPanel {
  private placingKind: string | null = null;
  private spawnBtns = new Map<string, HTMLButtonElement>();
  private target = el("select", { "aria-label": "소문을 들을 주민" });
  private about = el("select", { "aria-label": "소문의 대상" });

  constructor(private root: HTMLElement, private store: WorldStore, private actions: GodActions) {
    const weather = el("div", { class: "row" });
    for (const [w, label] of [["sunny", "☀️ 맑음"], ["overcast", "☁️ 흐림"], ["rain", "🌧️ 비"]]) {
      const b = el("button", { class: "btn" }, label);
      b.addEventListener("click", () => actions.send({ cmd: "god_weather", weather: w }));
      weather.append(b);
    }
    const spawn = el("div", { class: "row" });
    for (const [k, label] of SPAWNS) {
      const b = el("button", { class: "btn" }, label);
      b.addEventListener("click", () => this.setPlacing(this.placingKind === k ? null : k));
      this.spawnBtns.set(k, b);
      spawn.append(b);
    }
    const eventInput = el("input", { type: "text", placeholder: "예) 광장에 축제가 열렸다!", maxlength: "200" });
    const eventBtn = el("button", { class: "btn" }, "발생");
    const fireEvent = () => {
      if (!eventInput.value.trim()) return;
      actions.send({ cmd: "god_event", text: eventInput.value.trim() });
      eventInput.value = "";
    };
    eventBtn.addEventListener("click", fireEvent);
    eventInput.addEventListener("keydown", (e) => e.key === "Enter" && fireEvent());
    const rumorInput = el("input", { type: "text", placeholder: "예) 밤에 창고를 뒤졌대", maxlength: "200" });
    const rumorBtn = el("button", { class: "btn" }, "속삭이기");
    rumorBtn.addEventListener("click", () => {
      if (!rumorInput.value.trim() || this.target.value === this.about.value) return;
      actions.send({ cmd: "god_rumor", target: this.target.value, about: this.about.value, text: rumorInput.value.trim() });
      rumorInput.value = "";
    });
    const close = el("button", { class: "btn close", "aria-label": "닫기" }, "✕");
    close.addEventListener("click", () => this.toggle(false));
    root.append(
      close,
      el("h2", {}, "✨ 신 모드"),
      el("h3", {}, "날씨"),
      weather,
      el("h3", {}, "자원 생성 (선택 후 지도 클릭)"),
      spawn,
      el("h3", {}, "마을 사건"),
      el("div", { class: "row" }, eventInput, eventBtn),
      el("h3", {}, "소문 속삭이기"),
      el("div", { class: "row" }, this.target, el("span", {}, "에게"), this.about, el("span", {}, "에 대해")),
      el("div", { class: "row" }, rumorInput, rumorBtn),
    );
  }

  syncVillagers(): void {
    const opts = [...this.store.actors.values()].map((a) => [a.actor_id, a.name] as const);
    for (const sel of [this.target, this.about]) {
      const prev = sel.value;
      sel.replaceChildren(...opts.map(([id, name]) => el("option", { value: id }, name)));
      if (prev) sel.value = prev;
    }
    if (this.about.selectedIndex === this.target.selectedIndex && opts.length > 1) this.about.selectedIndex = 1;
  }

  get placing(): string | null {
    return this.placingKind;
  }

  setPlacing(kind: string | null): void {
    this.placingKind = kind;
    for (const [k, b] of this.spawnBtns) b.classList.toggle("on", k === kind);
    this.actions.setPlacing(kind);
  }

  toggle(force?: boolean): void {
    const show = force ?? this.root.classList.contains("hidden");
    this.root.classList.toggle("hidden", !show);
    if (!show) this.setPlacing(null);
  }
}

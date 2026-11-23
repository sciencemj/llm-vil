import { el, fmtClock, toast } from "./util";

interface Checkpoint {
  id: number;
  tick: number;
  day: number;
  minute: number;
  label: string;
  created_at: string;
}

interface SaveInfo {
  save_path: string | null;
  persistent: boolean;
  current_tick: number;
  current_day: number;
  current_minute: number;
  checkpoints: Checkpoint[];
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: (T & { detail?: unknown }) | null = null;
  try {
    data = text ? (JSON.parse(text) as T & { detail?: unknown }) : null;
  } catch {
    data = null;
  }
  if (!res.ok || data === null) {
    const detail = data && typeof data.detail === "string" ? data.detail : null;
    throw new Error(detail ?? `서버 오류 (HTTP ${res.status})`);
  }
  return data;
}

/** 세이브 관리: named checkpoints, restore (rewind), reset. Restores arrive over /ws as a
 *  fresh full snapshot, so the canvas and HUD update by themselves. */
export class SavesPanel {
  private info = el("div", { class: "persona" }, "불러오는 중…");
  private label = el("input", { type: "text", maxlength: "80", placeholder: "체크포인트 이름 (선택)" });
  private list = el("ul", { class: "list saves" });
  private result = el("div", { class: "result" });
  private resetBtn = el("button", { class: "btn danger" }, "🌱 새 마을 초기화");
  private confirmReset = false;
  private confirmTimer = 0;
  private clockTimer = 0;
  private busy = false;
  onOpen: () => void = () => {};
  /** Called before a restore/reset so replay mode lets the fresh live snapshot through. */
  beforeRewind: () => Promise<void> = async () => {};

  constructor(private root: HTMLElement, private clock: () => { day: number; minute: number }) {
    const close = el("button", { class: "btn close", "aria-label": "닫기" }, "✕");
    close.addEventListener("click", () => this.toggle(false));
    const save = el("button", { class: "btn on" }, "💾 체크포인트 저장");
    save.addEventListener("click", () => void this.save());
    this.label.addEventListener("keydown", (e) => e.key === "Enter" && void this.save());
    this.resetBtn.addEventListener("click", () => void this.reset());
    root.append(
      close,
      el("h2", {}, "💾 세이브 관리"),
      this.info,
      el("h3", {}, "새 체크포인트"),
      el("div", { class: "row" }, this.label, save),
      this.result,
      el("h3", {}, "저장된 체크포인트"),
      el("div", { class: "hint" }, "불러오면 그 시점 이후의 체크포인트와 기록은 지워져요"),
      this.list,
      el("h3", {}, "초기화"),
      el("div", { class: "hint" }, "1일차부터 새로 시작해요 (모든 체크포인트도 지워져요)"),
      el("div", { class: "row" }, this.resetBtn),
    );
  }

  get open(): boolean {
    return !this.root.classList.contains("hidden");
  }

  toggle(force?: boolean): void {
    const show = force ?? !this.open;
    this.root.classList.toggle("hidden", !show);
    this.setConfirm(false);
    window.clearInterval(this.clockTimer);
    if (show) {
      this.onOpen();
      void this.refresh();
      // keep the "현재 N일차 HH:MM" header live while the drawer is open
      this.clockTimer = window.setInterval(() => this.renderHeader(), 500);
    }
  }

  async refresh(): Promise<void> {
    try {
      this.render(await call<SaveInfo>("/api/saves/checkpoints"));
    } catch (err) {
      this.info.textContent = `세이브 정보를 불러오지 못했어요: ${(err as Error).message}`;
    }
  }

  private where = "";

  private renderHeader(): void {
    const c = this.clock();
    this.info.textContent = `현재 ${c.day + 1}일차 ${fmtClock(c.minute)}${this.where ? ` · ${this.where}` : ""}`;
  }

  private render(s: SaveInfo): void {
    this.where = s.persistent ? `저장 파일: ${s.save_path}` : "메모리 전용 실행 (저장 안 됨)";
    this.renderHeader();
    if (!s.checkpoints.length) {
      this.list.replaceChildren(el("li", {}, s.persistent ? "아직 체크포인트가 없어요" : "체크포인트를 쓰려면 저장 파일로 실행하세요"));
      return;
    }
    this.list.replaceChildren(
      ...s.checkpoints.map((c) => {
        const load = el("button", { class: "btn" }, "불러오기");
        load.addEventListener("click", () => void this.restore(c));
        return el(
          "li",
          { class: "save-row" },
          el(
            "div",
            {},
            el("b", {}, `${c.day + 1}일차 ${fmtClock(c.minute)}`),
            c.label ? el("span", {}, ` · ${c.label}`) : null,
            el("div", { class: "k" }, `#${c.id} · ${c.created_at}`),
          ),
          load,
        );
      }),
    );
  }

  private async run(work: () => Promise<string>): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    this.result.className = "result";
    this.result.textContent = "처리 중…";
    try {
      const msg = await work();
      this.result.className = "result ok";
      this.result.textContent = msg;
      toast(msg);
      await this.refresh();
    } catch (err) {
      this.result.className = "result fail";
      this.result.textContent = `✗ ${(err as Error).message}`;
    } finally {
      this.busy = false;
    }
  }

  private save(): Promise<void> {
    return this.run(async () => {
      const r = await call<{ created: Checkpoint }>("/api/saves/checkpoint", { label: this.label.value.trim() });
      this.label.value = "";
      return `✓ ${r.created.day + 1}일차 ${fmtClock(r.created.minute)} 저장됨`;
    });
  }

  private restore(c: Checkpoint): Promise<void> {
    return this.run(async () => {
      await this.beforeRewind();
      await call("/api/saves/restore", { id: c.id });
      return `⏪ ${c.day + 1}일차 ${fmtClock(c.minute)}${c.label ? ` (${c.label})` : ""}로 되돌렸어요`;
    });
  }

  private setConfirm(on: boolean): void {
    window.clearTimeout(this.confirmTimer); // never let an old timer flip a fresh state
    this.confirmReset = on;
    this.resetBtn.textContent = on ? "⚠️ 정말 초기화할까요? (다시 클릭)" : "🌱 새 마을 초기화";
    this.resetBtn.classList.toggle("on", on);
  }

  private async reset(): Promise<void> {
    if (!this.confirmReset) {
      this.setConfirm(true); // two-step confirm
      this.confirmTimer = window.setTimeout(() => this.setConfirm(false), 4000);
      return;
    }
    this.setConfirm(false);
    await this.run(async () => {
      await this.beforeRewind();
      await call("/api/saves/reset", {});
      return "🌱 새 마을로 초기화했어요";
    });
  }
}

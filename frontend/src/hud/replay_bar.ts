import type { WorldStore } from "../state/store";
import type { Snapshot } from "../state/types";
import { el, fmtClock, toast } from "./util";

interface ReplayRange {
  min_tick: number | null;
  max_tick: number | null;
  count: number;
  live_tick: number;
}

type ReplayFrame = Snapshot & { replay?: boolean; actors: (Snapshot["actors"][number] & { speech?: string | null })[] };

const SPEEDS = [1, 2, 4, 8];
const PLAY_MS = 250;
const TPD = 1440;

const label = (tick: number) => `${Math.floor(tick / TPD) + 1}일차 ${fmtClock(tick % TPD)}`;

/**
 * 🎞 Replay bar (`hud/replay_bar.ts`): scrub, step (±1 / ±10 sim minutes) and auto-play the
 * frames recorded by the backend (`/api/replay/range`, `/api/replay/frame`).
 *
 * - While active, live WebSocket messages are not applied (the simulation keeps running);
 *   "🔴 라이브 복귀" re-syncs from `/api/snapshot`.
 * - Frame fetches are coalesced: one request in flight, a newer target aborts the older one.
 * - `sessionSeq` makes enter/exit race-free: a slow exit can't clobber a newer session.
 * - Auto-play is single-flight: a step never starts while the previous one is awaiting.
 */
export class ReplayBar {
  active = false;
  private range: ReplayRange | null = null;
  private tick = 0;
  private slider = el("input", { type: "range", min: "0", max: "0", step: "1", "aria-label": "리플레이 위치" });
  private when = el("span", { class: "rlabel" }, "-");
  private playBtn = el("button", { class: "btn" }, "▶ 재생");
  private speedSel = el("select", { "aria-label": "리플레이 속도" });
  private playTimer = 0;
  private playAcc = 0;
  private abort: AbortController | null = null;
  private reqSeq = 0;
  private pendingTick: number | null = null;
  private sessionSeq = 0;
  private stepping = false;
  onChange: (active: boolean) => void = () => {};

  constructor(private root: HTMLElement, private store: WorldStore) {
    for (const s of SPEEDS) this.speedSel.append(el("option", { value: String(s) }, `${s}x`));
    this.speedSel.value = "2";
    const btn = (text: string, title: string, fn: () => void) => {
      const b = el("button", { class: "btn", title }, text);
      b.addEventListener("click", fn);
      return b;
    };
    this.slider.addEventListener("input", () => {
      this.stopPlay();
      this.seek(Number(this.slider.value));
    });
    this.playBtn.addEventListener("click", () => (this.playTimer ? this.stopPlay() : this.startPlay()));
    root.append(
      el("span", { class: "badge rbadge" }, "🎞 리플레이"),
      btn("⏪ -10분", "10분 뒤로", () => this.nudge(-10)),
      btn("◀ -1분", "1분 뒤로", () => this.nudge(-1)),
      el("div", { class: "rslider" }, this.slider, this.when),
      btn("▶ +1분", "1분 앞으로", () => this.nudge(1)),
      btn("⏩ +10분", "10분 앞으로", () => this.nudge(10)),
      this.playBtn,
      this.speedSel,
      btn("🔴 라이브 복귀", "리플레이 종료, 실시간으로 돌아가기", () => void this.exit()),
    );
  }

  async toggle(): Promise<void> {
    if (this.active) await this.exit();
    else await this.enter();
  }

  async enter(): Promise<void> {
    const r = await this.fetchRange();
    if (!r || r.max_tick === null) {
      toast("아직 녹화된 장면이 없어요");
      return;
    }
    this.sessionSeq++;
    this.active = true;
    this.store.replaying = true;
    this.root.classList.remove("hidden");
    document.getElementById("app")?.classList.add("replaying");
    this.onChange(true);
    this.seek(r.max_tick);
  }

  async exit(): Promise<void> {
    if (!this.active) return;
    const session = ++this.sessionSeq;
    this.stopPlay();
    this.abort?.abort();
    this.active = false;
    this.root.classList.add("hidden");
    document.getElementById("app")?.classList.remove("replaying");
    try {
      const res = await fetch("/api/snapshot");
      if (res.ok && session === this.sessionSeq) {
        const snap = (await res.json()) as Snapshot;
        if (session === this.sessionSeq) {
          this.store.replaying = false;
          this.store.apply(snap);
        }
      }
    } catch {
      // network hiccup: live deltas still resume below
    } finally {
      if (session === this.sessionSeq) {
        // only if the user did not re-enter replay while the snapshot was loading
        this.store.replaying = false;
        this.onChange(false);
      }
    }
  }

  /** Space bar while replaying: play / pause the recording. */
  togglePlay(): void {
    if (!this.active) return;
    if (this.playTimer) this.stopPlay();
    else this.startPlay();
  }

  private async fetchRange(): Promise<ReplayRange | null> {
    try {
      const res = await fetch("/api/replay/range");
      if (!res.ok) return null;
      const r = (await res.json()) as ReplayRange;
      this.range = r;
      if (r.min_tick !== null && r.max_tick !== null) {
        this.slider.min = String(r.min_tick);
        this.slider.max = String(r.max_tick);
      }
      return r;
    } catch {
      return null;
    }
  }

  private nudge(delta: number): void {
    this.stopPlay();
    this.seek(this.tick + delta);
  }

  /** Move to ``tick`` (clamped). The label updates instantly; the frame fetch is coalesced. */
  private seek(tick: number): void {
    const r = this.range;
    if (!r || r.min_tick === null || r.max_tick === null) return;
    this.tick = Math.max(r.min_tick, Math.min(r.max_tick, Math.round(tick)));
    this.slider.value = String(this.tick);
    this.when.textContent = label(this.tick);
    this.pendingTick = this.tick;
    if (!this.abort) void this.pump();
  }

  private async pump(): Promise<void> {
    while (this.pendingTick !== null && this.active) {
      const target = this.pendingTick;
      this.pendingTick = null;
      const seq = ++this.reqSeq;
      this.abort = new AbortController();
      try {
        const res = await fetch(`/api/replay/frame?tick=${target}`, { signal: this.abort.signal });
        if (!res.ok) continue;
        const frame = (await res.json()) as ReplayFrame;
        if (seq !== this.reqSeq || !this.active) continue; // a newer target won
        this.store.replaying = false; // let this one snapshot through
        this.store.apply(frame);
        this.store.replaying = true;
        const now = performance.now();
        for (const a of frame.actors) {
          const view = this.store.actors.get(a.actor_id);
          if (view) view.bubble = a.speech ? { text: a.speech, kind: "speech", until: now + 2500 } : null;
        }
      } catch {
        // aborted or network hiccup: the next seek retries
      } finally {
        this.abort = null;
      }
    }
  }

  private startPlay(): void {
    this.playBtn.textContent = "⏸ 정지";
    this.store.replayCondensed = Number(this.speedSel.value) >= 4;
    this.playAcc = 0;
    this.playTimer = window.setInterval(() => void this.playStep(), PLAY_MS);
  }

  private stopPlay(): void {
    window.clearInterval(this.playTimer);
    this.playTimer = 0;
    this.playBtn.textContent = "▶ 재생";
    this.store.replayCondensed = false;
  }

  private async playStep(): Promise<void> {
    if (this.stepping || !this.playTimer || !this.active) return; // single-flight
    this.stepping = true;
    try {
      this.store.replayCondensed = Number(this.speedSel.value) >= 4;
      // 1x = 2 sim minutes per real second, same as the live clock
      this.playAcc += Number(this.speedSel.value) * 2 * (PLAY_MS / 1000);
      const adv = Math.floor(this.playAcc);
      if (adv < 1) return;
      this.playAcc -= adv;
      const max = this.range?.max_tick ?? this.tick;
      if (this.tick + adv > max) {
        const r = await this.fetchRange(); // the live village keeps recording meanwhile
        if (!this.playTimer || !this.active) return; // stopped / exited while awaiting
        if (!r || r.max_tick === null || this.tick >= r.max_tick) {
          this.stopPlay();
          toast("녹화의 끝에 도달했어요");
          return;
        }
      }
      this.seek(this.tick + adv);
    } finally {
      this.stepping = false;
    }
  }
}

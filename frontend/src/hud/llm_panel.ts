import type { WorldStore } from "../state/store";
import { LLMSettings } from "./llm_settings";
import { el, fmtTick } from "./util";

interface CallRecord {
  tick: number;
  agent_id: string;
  purpose: string;
  tier: string;
  model: string;
  status: "ok" | "repaired" | "fallback";
  attempts: number;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  tokens_cached: number;
  error: string;
}

interface AgentLLM {
  calls: number;
  tokens_in: number;
  tokens_out: number;
  fallbacks: number;
  repaired: number;
  cost_usd: number;
  latency_ms_total: number;
}

export interface LLMStats {
  profile: string;
  provider: string;
  fast_model: string;
  smart_model: string;
  structured: string;
  base_url?: string | null;
  embed_profile: string;
  embed_provider: string;
  embed_model: string;
  embed_dim: number;
  embed_cache_hits: number;
  embed_cache_misses: number;
  embed_cache_hit_rate: number;
  total_calls: number;
  attempts: number;
  valid_first_try: number;
  repaired: number;
  fallbacks: number;
  timeouts: number;
  errors: number;
  validity_rate: number;
  tokens_in: number;
  tokens_out: number;
  tokens_cached: number;
  estimated_cost_usd: number;
  latency_ms: { p50: number; p95: number; max: number };
  in_flight: number;
  cognition_in_flight: number;
  per_agent: Record<string, AgentLLM>;
  recent_calls: CallRecord[];
}

const PURPOSE_KO: Record<string, string> = {
  choose_routine: "할 일 선택",
  plan_day: "하루 계획",
  converse_turn: "대화",
  reflect: "성찰",
  weekly_goals: "주간 목표",
};
const STATUS_KO: Record<CallRecord["status"], string> = { ok: "OK", repaired: "수리됨", fallback: "대체" };
const POLL_MS = 1500;

const num = (n: number) => n.toLocaleString("ko-KR");
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

function stat(label: string, value: string, hint = ""): HTMLElement {
  return el("div", { class: "stat", title: hint }, el("span", { class: "k" }, label), el("b", {}, value));
}

/** Compact LLM observability drawer. Polls only while open; renders only on change. */
export class LLMPanel {
  private timer = 0;
  private lastJson = "";
  private body = el("div");
  private status = el("div", { class: "persona" }, "불러오는 중…");
  private statusView = el("div", {}, this.status, this.body);
  private settingsView = el("div", { class: "hidden" });
  private settings = new LLMSettings(this.settingsView);
  private tabBtns = new Map<"status" | "settings", HTMLButtonElement>();
  private tab: "status" | "settings" = "status";
  onOpen: () => void = () => {};

  constructor(private root: HTMLElement, private store: WorldStore) {
    const close = el("button", { class: "btn close", "aria-label": "닫기" }, "✕");
    close.addEventListener("click", () => this.toggle(false));
    const tabs = el("div", { class: "tabs", role: "tablist" });
    for (const [key, label] of [["status", "현황"], ["settings", "설정"]] as const) {
      const b = el("button", { class: "btn tab", role: "tab" }, label);
      b.addEventListener("click", () => this.showTab(key));
      this.tabBtns.set(key, b);
      tabs.append(b);
    }
    this.settings.onApplied = () => {
      this.lastJson = "";
      void this.refresh();
    };
    root.append(close, el("h2", {}, "📊 LLM"), tabs, this.statusView, this.settingsView);
    this.showTab("status");
  }

  private showTab(tab: "status" | "settings"): void {
    this.tab = tab;
    for (const [k, b] of this.tabBtns) {
      b.classList.toggle("on", k === tab);
      b.setAttribute("aria-selected", String(k === tab));
    }
    this.statusView.classList.toggle("hidden", tab !== "status");
    this.settingsView.classList.toggle("hidden", tab !== "settings");
    window.clearInterval(this.timer);
    if (!this.open) return;
    if (tab === "status") {
      this.lastJson = "";
      void this.refresh();
      this.timer = window.setInterval(() => void this.refresh(), POLL_MS);
    } else {
      void this.settings.load();
    }
  }

  get open(): boolean {
    return !this.root.classList.contains("hidden");
  }

  toggle(force?: boolean): void {
    const show = force ?? !this.open;
    this.root.classList.toggle("hidden", !show);
    window.clearInterval(this.timer);
    if (show) {
      this.onOpen();
      this.showTab(this.tab);
    }
  }

  async refresh(): Promise<void> {
    try {
      const res = await fetch("/api/llm/stats");
      if (!res.ok) {
        this.status.textContent = `통계를 불러오지 못했어요 (HTTP ${res.status})`;
        return;
      }
      const json = await res.text();
      if (!this.open) return; // closed while the request was in flight
      if (json === this.lastJson) return;
      this.lastJson = json;
      this.status.textContent = "";
      this.render(JSON.parse(json) as LLMStats);
    } catch {
      this.status.textContent = "서버에 연결되지 않았어요";
    }
  }

  private render(s: LLMStats): void {
    const scroll = this.root.scrollTop;
    const models = s.fast_model === s.smart_model ? s.fast_model : `${s.fast_model} / ${s.smart_model}`;
    const agents = Object.entries(s.per_agent).sort((a, b) => b[1].tokens_in + b[1].tokens_out - (a[1].tokens_in + a[1].tokens_out));
    const maxTok = Math.max(1, ...agents.map(([, a]) => a.tokens_in + a.tokens_out));
    this.body.replaceChildren(
      el("h3", {}, "프로필"),
      el(
        "div",
        { class: "chips" },
        el("span", {}, `🧩 ${s.profile}`),
        el("span", {}, `🔌 ${s.provider}`),
        el("span", {}, `📐 ${s.structured}`),
      ),
      el("div", { class: "persona" }, `모델: ${models}`),
      el(
        "div",
        { class: "persona" },
        `임베딩: ${s.embed_provider} · ${s.embed_model} (${s.embed_dim}차원) · 캐시 적중 ${pct(s.embed_cache_hit_rate)} (${num(s.embed_cache_hits)}/${num(s.embed_cache_hits + s.embed_cache_misses)})`,
      ),
      el("h3", {}, "요약"),
      el(
        "div",
        { class: "stats" },
        stat("호출", num(s.total_calls), `시도 ${num(s.attempts)}회`),
        stat("JSON 유효율", pct(s.validity_rate)),
        stat("수리 / 대체", `${num(s.repaired)} / ${num(s.fallbacks)}`, `시간초과 ${s.timeouts}, 오류 ${s.errors}`),
        stat("진행 중", `${s.in_flight} · 생각 ${s.cognition_in_flight}`),
        stat("토큰 입력", num(s.tokens_in), `캐시 ${num(s.tokens_cached)}`),
        stat("토큰 출력", num(s.tokens_out)),
        stat("예상 비용", `$${s.estimated_cost_usd.toFixed(4)}`),
        stat("지연 p50/p95", `${s.latency_ms.p50.toFixed(0)} / ${s.latency_ms.p95.toFixed(0)}ms`, `최대 ${s.latency_ms.max.toFixed(0)}ms`),
      ),
      el("h3", {}, "주민별"),
      el(
        "ul",
        { class: "list" },
        ...agents.map(([id, a]) => {
          const tok = a.tokens_in + a.tokens_out;
          const fill = el("div", { class: "fill" });
          fill.style.cssText = `width:${(tok / maxTok) * 100}%;background:#5b8fd1`;
          return el(
            "li",
            {},
            el(
              "div",
              { class: "bar" },
              el("span", {}, this.store.name_of(id)),
              el("div", { class: "track" }, fill),
              el("span", {}, `${a.calls}회`),
            ),
            el("span", { class: "k" }, `토큰 ${num(tok)} · 수리 ${a.repaired} · 대체 ${a.fallbacks} · 평균 ${(a.latency_ms_total / Math.max(1, a.calls)).toFixed(0)}ms`),
          );
        }),
      ),
      el("h3", {}, "최근 호출"),
      el(
        "ul",
        { class: "list calls" },
        ...s.recent_calls.map((c) =>
          el(
            "li",
            {},
            el("span", { class: `badge st-${c.status}` }, STATUS_KO[c.status] ?? c.status),
            el("span", { class: "k" }, ` ${fmtTick(c.tick)} `),
            `${this.store.name_of(c.agent_id)} · ${PURPOSE_KO[c.purpose] ?? c.purpose} · ${c.tier === "deep" ? "smart" : "fast"}`,
            el("span", { class: "k" }, ` · ${c.latency_ms.toFixed(0)}ms · ${num(c.tokens_in)}→${num(c.tokens_out)}tok${c.attempts > 1 ? ` · ${c.attempts}회 시도` : ""}`),
            c.error ? el("div", { class: "k err" }, c.error) : null,
          ),
        ),
      ),
    );
    this.root.scrollTop = scroll;
  }

  dispose(): void {
    window.clearInterval(this.timer);
  }
}

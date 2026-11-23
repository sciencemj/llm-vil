import type { WorldStore } from "../state/store";
import type { AgentDetail, RelationshipAxes } from "../state/types";
import { ACTIVITY_KO, el, fmtClock, fmtTick, ITEM_KO, ko, ROUTINE_KO } from "./util";

const MEMORY_KO: Record<string, string> = {
  observation: "관찰", conversation: "대화", hearsay: "소문", plan: "계획", summary: "요약",
  fact: "사실", external_event: "사건", reflection: "성찰", goal: "목표",
};
const NEED_COLORS: Record<string, string> = { stamina: "#6f9a4a", hunger: "#d9822b", mood: "#5b8fd1" };
const NEED_KO: Record<string, string> = { stamina: "기력", hunger: "배고픔", mood: "기분" };
const AXES: [keyof RelationshipAxes, string, boolean][] = [
  ["familiarity", "친밀", false],
  ["affection", "호감", true],
  ["trust", "신뢰", true],
  ["respect", "존중", true],
];

function bar(label: string, value: number, max: number, color: string, bipolar = false): HTMLElement {
  const track = el("div", { class: `track${bipolar ? " bipolar" : ""}` });
  const fill = el("div", { class: "fill" });
  if (bipolar) {
    const pct = Math.min(50, (Math.abs(value) / max) * 50);
    fill.style.cssText = `position:absolute;top:0;height:100%;width:${pct}%;${value >= 0 ? "left:50%" : `left:${50 - pct}%`};background:${value >= 0 ? color : "#c8574d"}`;
  } else {
    fill.style.cssText = `width:${Math.max(0, Math.min(100, (value / max) * 100))}%;background:${color}`;
  }
  track.append(fill);
  return el("div", { class: "bar" }, el("span", {}, label), track, el("span", {}, value.toFixed(max > 1 ? 0 : 2)));
}

/** Directed relationship graph: selected villager at the center, edges coloured by affection. */
function drawGraph(canvas: HTMLCanvasElement, store: WorldStore, selfId: string, rels: Record<string, RelationshipAxes>): void {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320;
  const h = canvas.clientHeight || 190;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const ids = Object.keys(rels);
  const cx = w / 2;
  const cy = h / 2;
  const NODE = 15;
  const SELF = 18;
  const RX = w / 2 - NODE - 4; // ellipse: use the panel's full width for side edges
  const RY = h / 2 - NODE - 4;
  const pos = new Map<string, [number, number]>();
  ids.forEach((id, i) => {
    const a = -Math.PI / 2 + (i / ids.length) * Math.PI * 2;
    pos.set(id, [cx + Math.cos(a) * RX, cy + Math.sin(a) * RY]);
  });
  for (const id of ids) {
    const r = rels[id];
    const [x, y] = pos.get(id) ?? [cx, cy];
    const back = store.relationships[id]?.[selfId];
    const aff = r.affection;
    ctx.strokeStyle = aff >= 0 ? `rgba(111, 154, 74, ${0.35 + aff * 0.65})` : `rgba(200, 87, 77, ${0.35 - aff * 0.65})`;
    ctx.lineWidth = 1 + r.familiarity * 6;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    // label sits mid-way between the two node rims, on a pill, never under a node
    const len = Math.hypot(x - cx, y - cy) || 1;
    const t = (SELF + (len - SELF - NODE) / 2) / len;
    const mx = cx + (x - cx) * t;
    const my = cy + (y - cy) * t;
    const sign = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
    const label = back ? `${sign(aff)}/${sign(back.affection)}` : sign(aff);
    ctx.font = "700 10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const lw = ctx.measureText(label).width + 6;
    ctx.fillStyle = "rgba(251, 246, 234, 0.92)";
    ctx.fillRect(mx - lw / 2, my - 7, lw, 14);
    ctx.fillStyle = "#5d533f";
    ctx.fillText(label, mx, my);
  }
  const node = (x: number, y: number, label: string, self: boolean) => {
    ctx.fillStyle = self ? "#d9822b" : "#fbf6ea";
    ctx.strokeStyle = "#2b2620";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, self ? SELF : NODE, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = self ? "#fff" : "#2b2620";
    ctx.font = "800 11px 'Apple SD Gothic Neo', 'Noto Sans KR', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y);
  };
  for (const id of ids) {
    const [x, y] = pos.get(id) ?? [cx, cy];
    node(x, y, store.name_of(id), false);
  }
  node(cx, cy, store.name_of(selfId), true);
}

export class Inspector {
  private current: string | null = null;
  private timer = 0;
  private reqSeq = 0;
  private lastJson = "";
  onClose: () => void = () => {};

  constructor(private root: HTMLElement, private store: WorldStore) {}

  get selected(): string | null {
    return this.current;
  }

  open(id: string): void {
    if (this.current !== id) this.lastJson = "";
    this.current = id;
    this.root.classList.remove("hidden");
    void this.refresh();
    window.clearInterval(this.timer);
    this.timer = window.setInterval(() => void this.refresh(), 2000);
  }

  close(): void {
    this.current = null;
    window.clearInterval(this.timer);
    this.root.classList.add("hidden");
    this.onClose();
  }

  async refresh(): Promise<void> {
    const id = this.current;
    if (!id) return;
    const seq = ++this.reqSeq;
    try {
      const res = await fetch(`/api/agent/${encodeURIComponent(id)}`);
      if (!res.ok) return;
      const d = (await res.json()) as AgentDetail;
      if (seq !== this.reqSeq || this.current !== id) return;
      this.store.updateRelationships(d.id, d.relationships, d.incoming ?? {});
      const json = JSON.stringify(d);
      if (json === this.lastJson) return; // nothing changed: keep DOM, focus and selection
      if (this.userIsSelecting()) return; // retry on the next poll
      this.lastJson = json;
      this.render(d);
    } catch {
      // server restarting; the next poll retries
    }
  }

  private targetName(id: string): string {
    return (
      this.store.objects.get(id)?.name ??
      this.store.areas.find((a) => a.id === id)?.name ??
      (id === "my_plots" ? "내 밭" : this.store.name_of(id))
    );
  }

  private userIsSelecting(): boolean {
    const sel = window.getSelection();
    const active = document.activeElement;
    return (
      (!!sel && !sel.isCollapsed && sel.anchorNode !== null && this.root.contains(sel.anchorNode)) ||
      (active !== null && active !== document.body && this.root.contains(active) && active.tagName !== "BUTTON")
    );
  }

  private render(d: AgentDetail): void {
    const scroll = this.root.scrollTop;
    const close = el("button", { class: "btn close", "aria-label": "닫기" }, "✕");
    close.addEventListener("click", () => this.close());
    const routine = d.routine
      ? `${ko(ROUTINE_KO, d.routine.routine)}${d.routine.target !== "none" ? ` → ${this.targetName(d.routine.target)}` : ""}`
      : "생각 중…";
    const graph = el("canvas", { id: "relgraph" });
    const rels = Object.entries(d.relationships).map(([tid, r]) =>
      el(
        "div",
        { class: "rel" },
        el("span", { class: "who" }, this.store.name_of(tid)),
        el("span", { class: "sum" }, r.summary.replace(/^[^:]+:\s*/, "")),
        ...AXES.map(([k, label, bipolar]) => bar(label, r[k] as number, 1, "#6f9a4a", bipolar)),
      ),
    );
    const plan = d.plan.map((b) =>
      el("li", {}, el("span", { class: "k" }, `${fmtClock(b.start_minute)}–${fmtClock(b.end_minute)}`), `${ko(ROUTINE_KO, b.routine)} · ${b.note}`),
    );
    this.root.replaceChildren(
      close,
      el("h2", {}, `${d.name}`),
      el("div", { class: "persona" }, `${d.age}세 · ${d.occupation} · ${d.in_conversation ? "💬 대화 중" : ko(ACTIVITY_KO, d.activity)}`),
      el("div", { class: "traits" }, ...d.traits.map((t) => el("span", {}, t))),
      el("div", { class: "persona" }, d.persona),
      el("h3", {}, "지금 하는 일"),
      el("div", { class: "persona" }, routine, d.routine?.reason ? el("span", { class: "sum" }, ` (${d.routine.reason})`) : null),
      el("h3", {}, "상태"),
      ...(["stamina", "hunger", "mood"] as const).map((k) => bar(NEED_KO[k], d.needs[k], 100, NEED_COLORS[k])),
      el("h3", {}, "소지품"),
      el("div", { class: "chips" }, ...Object.entries(d.inventory).map(([k, v]) => el("span", {}, `${ko(ITEM_KO, k)} ×${v}`))),
      el("h3", {}, "삶의 목표"),
      el(
        "ul",
        { class: "list" },
        ...(d.goal_details && d.goal_details.length
          ? d.goal_details.map((g) =>
              el("li", {}, `🎯 ${g.text}`, el("span", { class: "k" }, ` · 근거 기억 ${g.evidence_count}개 · ${fmtTick(g.tick)}`)),
            )
          : d.goals.length
            ? d.goals.map((g) => el("li", {}, `🎯 ${g}`))
            : [el("li", {}, "아직 없음 (주간 성찰 후 생김)")]),
      ),
      el("h3", {}, "소문"),
      el(
        "ul",
        { class: "list rumors" },
        ...(d.rumors && d.rumors.length
          ? d.rumors.map((r) =>
              el(
                "li",
                {},
                el("span", { class: "k" }, `${fmtTick(r.tick)} · ${r.source}에게서 · 확신 ${Math.round(r.confidence * 100)}%`),
                el("div", {}, `🗣️ ${r.text}`),
                el("div", { class: "conf" }, el("div", { class: "fill", style: `width:${Math.round(r.confidence * 100)}%` })),
              ),
            )
          : [el("li", {}, "들은 소문이 없어요")]),
      ),
      el("h3", {}, "관계 그래프"),
      graph,
      ...rels,
      el("h3", {}, "오늘의 계획"),
      el("ul", { class: "list" }, ...plan),
      el("h3", {}, "성찰"),
      el("ul", { class: "list" }, ...d.reflections.map((r) => el("li", {}, el("span", { class: "k" }, fmtTick(r.tick)), r.text))),
      el("h3", {}, "최근 기억"),
      el("ul", { class: "list" }, ...d.recent_memories.map((m) => el("li", {}, el("span", { class: "k" }, `${fmtTick(m.tick)} · ${ko(MEMORY_KO, m.kind)} · 중요도 ${m.importance}`), m.text))),
      el("h3", {}, "메모"),
      el("ul", { class: "list" }, ...d.world_notes.map((n) => el("li", {}, n))),
    );
    drawGraph(graph, this.store, d.id, d.relationships);
    this.root.scrollTop = scroll;
  }
}

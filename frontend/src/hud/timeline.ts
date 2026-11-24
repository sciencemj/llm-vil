import type { WorldStore } from "../state/store";
import type { TimelineEvent } from "../state/types";
import { ACTIVITY_KO, el, fmtTick, ITEM_KO, MONSTER_KO, WEATHER_KO, ko } from "./util";

const FAIL_KO: Record<string, string> = {
  state_stump: "이미 베어진 나무", state_empty: "이미 딴 덤불", state_tilled: "이미 갈린 밭",
  too_far: "너무 멂", too_tired: "기력 부족", forest_reserve: "숲 보호 규칙", chest_empty: "상자가 빔",
  no_item: "물건 없음", busy: "다른 일 중", blocked: "길이 막힘", merchant_absent: "상인 없음",
};

type Filter = "all" | "dialogue" | "events" | "god";

export function describe(ev: TimelineEvent, store: WorldStore): { text: string; cls: string } {
  const who = store.name_of(ev.actor_id);
  const d = ev.data;
  const item = (k: unknown) => ko(ITEM_KO, k);
  switch (ev.kind) {
    case "utterance":
      return { text: `${who} → ${store.name_of(ev.target_id)}: “${String(d.text ?? "")}”`, cls: "dialogue" };
    case "conversation_ended":
      return { text: `${who}와 ${store.name_of(ev.target_id)}의 대화가 끝났다 (${String(d.topic ?? "")})`, cls: "dialogue" };
    case "harvested":
      return { text: `${who}이(가) ${item(d.crop)}을 수확했다`, cls: "events" };
    case "chopped":
      return { text: `${who}이(가) 나무를 베었다`, cls: "events" };
    case "foraged":
      return { text: `${who}이(가) 산딸기를 땄다`, cls: "events" };
    case "cooked":
      return { text: `${who}이(가) ${item(d.recipe)}를 만들었다`, cls: "events" };
    case "crafted":
      return { text: `${who}이(가) ${item(d.output)} ${String(d.qty ?? 1)}개를 제작했다`, cls: "events" };
    case "mined":
      return { text: `${who}이(가) 채석장에서 광석을 캤다`, cls: "events" };
    case "gathered":
      return { text: `${who}이(가) 약초를 채집했다`, cls: "events" };
    case "shop_sale":
      return {
        text: `${who}이(가) ${store.name_of(ev.target_id)} 가게에서 ${item(d.item)} ${String(d.qty ?? 1)}개를 샀다 (${String(d.total ?? d.price ?? "")}🪙)`,
        cls: "events",
      };
    case "shop_stocked":
      return { text: `${who}이(가) 가게에 ${item(d.item)} ${String(d.qty ?? 1)}개를 진열했다`, cls: "events" };
    case "p2p_trade":
      return {
        text: `${who} ↔ ${store.name_of(ev.target_id)} 직거래: ${item(d.give_item)} ×${String(d.give_qty)} ↔ ${item(d.want_item)} ×${String(d.want_qty)}`,
        cls: "dialogue",
      };
    case "drops_sold":
      return { text: `${who}이(가) 길드에 ${item(d.item)} ${String(d.qty ?? 1)}개를 매각했다 (+${String(d.coins)}🪙)`, cls: "events" };
    case "bounty_posted":
      return { text: `${who}이(가) ${ko(MONSTER_KO, d.item)} 현상금을 걸었다 (${String(d.reward)}🪙)`, cls: "events" };
    case "bounty_completed":
      return { text: `${who}이(가) ${ko(MONSTER_KO, d.species)} 현상금을 달성했다 (+${String(d.reward)}🪙)`, cls: "events" };
    case "request_posted":
      return { text: `${who}이(가) ${item(d.item)} ${String(d.qty ?? 1)}개 의뢰를 올렸다 (${String(d.reward)}🪙)`, cls: "events" };
    case "request_expired":
      return { text: `${who}의 의뢰가 만료되어 예치금 ${String(d.refund)}🪙가 환급되었다`, cls: "events" };
    case "monster_defeated":
      return { text: `⚔️ ${who}이(가) ${ko(MONSTER_KO, d.species)}을(를) 처치했다!`, cls: "events" };
    case "fight_lost":
      return { text: `${who}이(가) ${ko(MONSTER_KO, d.species)}와(과)의 전투에서 밀렸다`, cls: "events" };
    case "fainted":
      return { text: `💫 ${who}이(가) 기절해 치료소로 옮겨졌다`, cls: "events" };
    case "monster_respawned":
      return { text: `이끼숲에 ${ko(MONSTER_KO, d.species)}이(가) 다시 나타났다`, cls: "events" };
    case "tool_repaired":
      return { text: `${who}이(가) 대장간에서 ${item(d.item)}을(를) 수리했다`, cls: "events" };
    case "price_set":
      return { text: `${who}이(가) ${item(d.item)} 가격을 ${String(d.price)}🪙로 정했다`, cls: "events" };
    case "tax_paid":
      return { text: `${who}이(가) 마을 세금 ${String(d.coins)}🪙를 납부했다`, cls: "events" };
    case "ate":
      return { text: `${who}이(가) ${item(d.item)}을 먹었다`, cls: "events" };
    case "gift":
      return { text: `${who}이(가) ${store.name_of(ev.target_id)}에게 ${item(d.item)}을 선물했다`, cls: "dialogue" };
    case "traded":
      return { text: `${who}이(가) 장터에서 ${item(d.item)} ${String(d.qty)}개를 ${d.op === "buy" ? "샀다" : "팔았다"}`, cls: "events" };
    case "order_fulfilled":
      return { text: `${who}이(가) 게시판 주문을 해결했다 (+${String(d.reward)}동전)`, cls: "events" };
    case "chest_taken":
      return { text: `누군가 ${store.name_of(ev.target_id)}의 상자에서 물건을 가져갔다…`, cls: "events" };
    case "chest_missing_discovered":
      return { text: `${who}이(가) 상자에서 물건이 사라진 것을 알아챘다!`, cls: "events" };
    case "merchant_arrived":
      return { text: "떠돌이 상인이 장터에 도착했다", cls: "events" };
    case "merchant_left":
      return { text: "떠돌이 상인이 떠났다", cls: "events" };
    case "weather":
    case "weather_changed":
      return { text: `오늘의 날씨: ${ko(WEATHER_KO, d.weather)}`, cls: ev.kind === "weather" ? "events" : "god" };
    case "stored":
      return { text: `${who}이(가) 상자에 ${item(d.item)} ${String(d.qty ?? 1)}개를 넣었다`, cls: "events" };
    case "took":
      return { text: `${who}이(가) 상자에서 ${item(d.item)} ${String(d.qty ?? 1)}개를 꺼냈다`, cls: "events" };
    case "planted":
      return { text: `${who}이(가) ${item(d.crop)} 씨앗을 심었다`, cls: "events" };
    case "tilled":
      return { text: `${who}이(가) 밭을 갈았다`, cls: "events" };
    case "watered":
      return { text: `${who}이(가) 밭에 물을 주었다`, cls: "events" };
    case "resource_spawned":
      return { text: `새 자원이 나타났다 (${String(d.object_kind ?? "")})`, cls: "god" };
    case "fell_asleep":
      return { text: `${who}이(가) 잠들었다`, cls: "events" };
    case "action_interrupted":
      return { text: `${who}이(가) 하던 일을 멈췄다 (${ko({ conversation: "대화", external_event: "마을 사건" }, d.reason)})`, cls: "events" };
    case "action_failed":
      return { text: `${who}의 ${ko(ACTIVITY_KO, d.action ?? "행동")} 시도가 실패했다 (${ko(FAIL_KO, d.reason)})`, cls: "events" };
    case "woke":
      return { text: `${who}이(가) 일어났다`, cls: "events" };
    default:
      if (ev.kind.startsWith("god_")) {
        const GOD_KO: Record<string, string> = {
          set_weather: "날씨 변경", spawn_resource: "자원 생성", inject_event: "마을 사건", inject_rumor: "소문",
        };
        const body = d.weather ? ko(WEATHER_KO, d.weather) : String(d.text ?? d.object_kind ?? "");
        return { text: `✨ 신의 개입 · ${ko(GOD_KO, ev.kind.slice(4))}: ${body}`, cls: "god" };
      }
      return { text: `${ev.actor_id ? who + ": " : ""}${ev.kind}`, cls: "events" };
  }
}

/** A raw world event that just repeats the God command logged right before it. */
function isGodEcho(ev: TimelineEvent, prev: TimelineEvent | undefined): boolean {
  if (!prev || prev.tick !== ev.tick) return false;
  return (
    (ev.kind === "weather_changed" && prev.kind === "god_set_weather") ||
    (ev.kind === "resource_spawned" && prev.kind === "god_spawn_resource")
  );
}

export class Timeline {
  private list = el("ul");
  private filter: Filter = "all";
  private rendered = -1;

  constructor(root: HTMLElement, private store: WorldStore) {
    const select = el("select", { "aria-label": "타임라인 필터" });
    for (const [v, label] of [["all", "전체"], ["dialogue", "대화"], ["events", "사건"], ["god", "신의 개입"]]) {
      select.append(el("option", { value: v }, label));
    }
    select.addEventListener("change", () => {
      this.filter = select.value as Filter;
      this.rendered = -1;
      this.render();
    });
    root.append(el("div", { class: "head" }, "📜 마을 소식", select), this.list);
  }

  render(): void {
    const tl = this.store.timeline;
    const sig = this.store.timelineSeq;
    if (sig === this.rendered) return;
    this.rendered = sig;
    const items = tl
      .filter((ev, i) => !isGodEcho(ev, tl[i - 1]))
      .map((ev) => ({ ev, ...describe(ev, this.store) }))
      .filter((x) => this.filter === "all" || x.cls === this.filter)
      .slice(-80)
      .reverse();
    this.list.replaceChildren(
      ...items.map((x) => el("li", { class: x.cls }, el("span", { class: "t" }, fmtTick(x.ev.tick)), el("span", { class: "msg" }, x.text))),
    );
  }
}

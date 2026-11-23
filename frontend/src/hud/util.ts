export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

export function fmtClock(minute: number): string {
  const h = Math.floor(minute / 60);
  const m = Math.floor(minute % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function fmtTick(tick: number, ticksPerDay = 1440): string {
  return `${Math.floor(tick / ticksPerDay) + 1}일 ${fmtClock(tick % ticksPerDay)}`;
}

let toastTimer = 0;
export function toast(msg: string): void {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.remove("hidden");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => t.classList.add("hidden"), 2200);
}

export const WEATHER_KO: Record<string, string> = { sunny: "☀️ 맑음", overcast: "☁️ 흐림", rain: "🌧️ 비" };

export const ITEM_KO: Record<string, string> = {
  wood: "장작",
  berry: "산딸기",
  turnip: "순무",
  wheat: "밀",
  turnip_seed: "순무씨",
  wheat_seed: "밀씨",
  berry_stew: "산딸기 스튜",
  turnip_soup: "순무 수프",
  bread: "빵",
  coins: "동전",
};

export const ROUTINE_KO: Record<string, string> = {
  tend_farm: "밭 가꾸기",
  chop_wood: "장작 패기",
  forage: "채집",
  cook_meal: "요리",
  eat_food: "식사",
  store_or_take: "상자 정리",
  visit_and_talk: "이웃 방문",
  post_or_fulfill_order: "게시판 주문",
  trade: "장터 거래",
  sleep: "잠",
  rest: "휴식",
  wander: "산책",
};

export const ACTIVITY_KO: Record<string, string> = {
  idle: "한가함",
  walking: "걷는 중",
  sleeping: "자는 중",
  resting: "쉬는 중",
  in_conversation: "대화 중",
  chop: "나무 베는 중",
  till: "밭 가는 중",
  plant: "씨 뿌리는 중",
  water: "물 주는 중",
  harvest: "수확 중",
  forage: "채집 중",
  cook: "요리 중",
  eat: "먹는 중",
  store: "상자에 넣는 중",
  take: "상자에서 꺼내는 중",
  trade: "거래 중",
  fulfill_order: "주문 처리 중",
  wait: "기다리는 중",
};

export const ko = (table: Record<string, string>, key: unknown): string => table[String(key)] ?? String(key ?? "");

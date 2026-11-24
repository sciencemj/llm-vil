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

export const PROFESSION_KO: Record<string, string> = {
  farmer: "🌾 농부",
  woodcutter: "🪓 나무꾼",
  chef_innkeeper: "🍲 요리사·여관주인",
  merchant: "🪙 상인",
  herbalist: "🌿 약초꾼·연금술사",
  baker: "🥖 빵집 사장",
  blacksmith: "⚒️ 대장장이",
  miner: "⛏️ 광부",
  adventurer: "⚔️ 모험가",
  guildmaster: "🛡️ 길드장",
};

export const EQUIP_SLOT_KO: Record<string, string> = {
  weapon: "무기",
  armor: "방어구",
  tool: "도구",
};

export const MONSTER_KO: Record<string, string> = {
  slime: "슬라임",
  boar: "멧돼지",
  goblin: "고블린",
};

export const ITEM_KO: Record<string, string> = {
  wood: "장작",
  charcoal: "숯",
  berry: "산딸기",
  turnip: "순무",
  wheat: "밀",
  herb: "약초",
  turnip_seed: "순무씨",
  wheat_seed: "밀씨",
  berry_stew: "산딸기 스튜",
  turnip_soup: "순무 수프",
  roast_meat: "고기 구이",
  bread: "빵",
  heal_potion: "회복 물약",
  stamina_potion: "기력 물약",
  iron_ore: "철광석",
  coal: "석탄",
  stone: "돌",
  slime_gel: "슬라임 젤",
  boar_meat: "멧돼지 고기",
  boar_hide: "멧돼지 가죽",
  goblin_ore: "고블린 광석",
  axe: "도끼",
  pickaxe: "곡괭이",
  hoe: "괭이",
  sword: "검",
  armor: "갑옷",
  coins: "동전",
};

export const ROUTINE_KO: Record<string, string> = {
  tend_farm: "밭 가꾸기",
  chop_wood: "장작 패기",
  forage: "채집",
  cook_meal: "요리",
  craft: "제작",
  eat_food: "식사",
  use_potion: "물약 복용",
  store_or_take: "상자 정리",
  visit_and_talk: "이웃 방문",
  post_or_fulfill_order: "게시판 주문",
  post_request: "물품 의뢰",
  post_bounty: "현상금 공고",
  trade: "장터 거래",
  stock_shop: "가게 진열",
  buy_from_shop: "가게 쇼핑",
  run_stall: "가게 운영",
  mine: "채광",
  gather_herb: "약초 채집",
  hunt: "몬스터 사냥",
  sell_drops: "전리품 매각",
  repair_tool: "장비 수리",
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
  craft: "제작 중",
  mine: "채광 중",
  gather_herb: "약초 채집 중",
  fight: "전투 중",
  eat: "먹는 중",
  store: "상자에 넣는 중",
  take: "상자에서 꺼내는 중",
  trade: "거래 중",
  stock_shop: "진열 중",
  buy_from_shop: "구매 중",
  set_prices: "가격 조정 중",
  sell_drops: "전리품 정산 중",
  post_bounty: "현상금 공고 중",
  post_request: "의뢰 등록 중",
  repair_tool: "장비 수리 중",
  fulfill_order: "주문 처리 중",
  wait: "기다리는 중",
};

export const ko = (table: Record<string, string>, key: unknown): string => table[String(key)] ?? String(key ?? "");

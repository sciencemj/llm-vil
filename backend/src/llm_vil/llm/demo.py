"""Deterministic, zero-cost, rule-based ``DemoBackend``.

Reads the structured ``LLMRequest.context`` (never the prose prompt) and answers like a
Stardew-style villager. All randomness is ``stable_unit`` hashing of the context, so runs
are bit-identical; ``simulated_latency_ms`` injects real ``asyncio`` latency for race tests.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .protocol import HashingEmbedder, LLMRequest, LLMResponse, dumps, estimate_tokens, stable_unit

COOKED = ("roast_meat", "turnip_soup", "berry_stew", "bread")
RAW = ("turnip", "berry")
RECIPE_PREF = ("roast_meat", "turnip_soup", "berry_stew", "bread")
FOOD_BUY_PREF = ("bread", "turnip_soup", "berry_stew", "roast_meat", "turnip", "boar_meat",
                 "berry")
KEEP = {"turnip": 3, "wood": 2, "herb": 0, "wheat": 0, "berry": 3, "bread": 2,
        "turnip_soup": 1, "berry_stew": 1, "roast_meat": 1}
TOPICS = ("안부", "농사", "날씨", "장터", "마을 소문", "요리", "숲")
OPENERS = {
    "안부": "요즘 어떻게 지내?",
    "농사": "밭은 잘 되어 가?",
    "날씨": "오늘 날씨 어때 보여?",
    "장터": "상인 언제 또 온대?",
    "마을 소문": "혹시 요즘 들은 소식 있어?",
    "요리": "오늘 뭐 해 먹었어?",
    "숲": "숲에 나무 많이 남았어?",
}
REPLIES = (
    "응, 덕분에 그럭저럭 잘 지내고 있어.",
    "순무가 꽤 잘 자라고 있더라.",
    "비가 좀 와 주면 좋겠어.",
    "상인한테 씨앗을 좀 사 둬야겠어.",
    "다음에 같이 스튜 끓여 먹자.",
    "숲 쪽은 아직 나무가 많더라.",
    "너는 요즘 무슨 일 하면서 지내?",
    "그러게, 우리 마을도 많이 변했지.",
)
CLOSERS = ("그럼 이만 가볼게, 또 보자!", "얘기 즐거웠어. 다음에 또!", "나중에 또 이야기하자.")
WORK = {"farm": "tend_farm", "wood": "chop_wood", "forage": "forage"}


PROFESSION_WORK = {
    "farmer": "tend_farm", "woodcutter": "chop_wood", "chef_innkeeper": "cook_meal",
    "merchant": "stock_shop", "herbalist": "gather_herb", "baker": "craft",
    "blacksmith": "craft", "miner": "mine", "adventurer": "hunt", "guildmaster": "post_bounty",
}
COIN_CUSHION = 30    # villagers keep about this much; the rest circulates at the stall
LUXURY_COINS = 70    # above this, villagers treat themselves at other villagers' shops
PRODUCTION_CAP = 8   # producers pause a good once they hold this many unsold


class DemoBackend:
    model = "demo-rules-v1"

    def __init__(self, simulated_latency_ms: float = 0.0) -> None:
        self.simulated_latency_ms = simulated_latency_ms
        self._embedder = HashingEmbedder()
        self.calls = 0

    async def complete_json(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        await asyncio.sleep(self.simulated_latency_ms / 1000 if self.simulated_latency_ms else 0)
        self.calls += 1
        handler = getattr(self, f"_{req.purpose}")
        text = dumps(handler(req.context))
        return LLMResponse(
            text=text, model=self.model, tokens_in=estimate_tokens(req.prompt_text()),
            tokens_out=estimate_tokens(text), latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed(t) for t in texts]

    # ------------------------------------------------------------------ plan_day
    def _probe(self, c: dict[str, Any]) -> dict[str, Any]:
        """Connection test used by the provider settings UI / CLI."""
        return {"ok": True}

    def _plan_day(self, c: dict[str, Any]) -> dict[str, Any]:
        s = c["self"]
        w, sl = s["wake_minute"], s["sleep_minute"]
        bias = s["work_bias"]
        ranked = sorted(bias, key=lambda k: (-bias[k], k))
        primary = PROFESSION_WORK.get(s.get("profession", ""), WORK[ranked[0]])
        secondary = primary if s.get("profession") in PROFESSION_WORK else WORK[
            ranked[1 if stable_unit(s["id"], c["day"], "plan") < 0.6 else 2]]
        plots_need = any(v in ("untilled", "tilled", "planted", "ready")
                         for v in c["plots"].values())
        first = "tend_farm" if plots_need else primary
        return {"blocks": [
            {"start_minute": w, "end_minute": w + 180, "routine": first, "note": "아침 일"},
            {"start_minute": w + 180, "end_minute": 690, "routine": primary, "note": "오전 일"},
            {"start_minute": 690, "end_minute": 780, "routine": "wander", "note": "점심·수다"},
            {"start_minute": 780, "end_minute": 1020, "routine": secondary, "note": "오후 일"},
            {"start_minute": 1020, "end_minute": sl, "routine": "wander", "note": "저녁 휴식"},
        ]}

    # ------------------------------------------------------------------ choose_routine
    def _choose_routine(self, c: dict[str, Any]) -> dict[str, Any]:
        s = c["self"]
        m, day = c["minute"], c["day"]
        opts: dict[str, dict[str, list[str]]] = c["options"]
        inv: dict[str, int] = c["inventory"]
        needs = c["needs"]
        th = c["thresholds"]

        def u(*salt: Any) -> float:
            return stable_unit(s["id"], day, m, c["reason"], *salt)

        def pick(routine: str, target: str | None = None, **kw: Any) -> dict[str, Any]:
            out = {"routine": routine,
                   "target": target if target is not None else opts[routine]["targets"][0],
                   "item": "none", "qty": 1, "op": "none", "topic": "", "facing_cue": "none",
                   "expression": "neutral"}
            out.update(kw)
            return out

        if m >= s["sleep_minute"] or m < s["wake_minute"] - 30:
            return pick("sleep", s["bed_id"], reason="잘 시간", expression="tired")

        eat_items = opts.get("eat_food", {}).get("items", [])
        if needs["hunger"] >= th["hungry"]:
            for food in COOKED:
                if food in eat_items:
                    return pick("eat_food", "none", item=food, reason="배고픔", expression="happy")
            if "cook_meal" in opts:
                items = opts["cook_meal"]["items"]
                recipe = next(r for r in RECIPE_PREF if r in items)
                return pick("cook_meal", item=recipe, reason="배고파서 요리")
            buy = self._buy_food(c, pick)
            if buy is not None:
                return buy
            for food in (*RAW, "boar_meat"):
                if food in eat_items:
                    return pick("eat_food", "none", item=food, reason="배고픔")
            if "forage" in opts:
                t = opts["forage"]["targets"]
                return pick("forage", t[int(u("f") * len(t))], reason="먹을 게 없음")
            take = self._take_food(c, u)
            if take is not None:
                return take
            if "trade" in opts and "bread" in opts["trade"]["items"]:
                return pick("trade", item="bread", op="buy", qty=1, reason="빵 사기")
            if "tend_farm" in opts:
                return pick("tend_farm", "my_plots", reason="작물을 키워야 함")
            return pick("wander", "forest", reason="먹을 것 찾기", expression="sad")

        if c.get("hp", 100) < 45 and "use_potion" in opts and (
            "heal_potion" in opts["use_potion"]["items"]
        ):
            return pick("use_potion", "none", item="heal_potion", reason="상처 치료")

        if needs["stamina"] < th["tired"]:
            if "use_potion" in opts and "stamina_potion" in opts["use_potion"]["items"] and (
                needs["stamina"] < 15
            ):
                return pick("use_potion", "none", item="stamina_potion", reason="기력 회복")
            return pick("rest", reason="지침", expression="tired")

        if "repair_tool" in opts:
            return pick("repair_tool", item=opts["repair_tool"]["items"][0],
                        reason="도구 수리", expression="neutral")

        has_cooked = any(f in eat_items for f in COOKED)
        if not has_cooked and "cook_meal" in opts and (600 <= m < 720 or 990 <= m < 1080):
            items = opts["cook_meal"]["items"]
            return pick("cook_meal", item=next(r for r in RECIPE_PREF if r in items),
                        reason="끼니 준비")

        food_held = sum(n for f, n in inv.items() if f in (*COOKED, *RAW, "boar_meat"))
        if food_held == 0 and needs["hunger"] >= 25:
            buy = self._buy_food(c, pick)
            if buy is not None:
                return buy

        if "post_or_fulfill_order" in opts:
            return pick("post_or_fulfill_order", item=opts["post_or_fulfill_order"]["items"][0],
                        reason="주문 해결")

        if inv.get("coins", 0) >= LUXURY_COINS and u("treat") < 0.3:
            treat = self._treat(c, pick)
            if treat is not None:
                return treat

        req = self._request_missing(c, pick, u)
        if req is not None:
            return req

        social_time = (690 <= m < 780 or 1020 <= m < 1140)
        if not social_time or u("work_first") < 0.5:
            job = self._profession(c, pick, u)
            if job is not None:
                return job

        prices = c.get("item_prices") or {}

        def price(item: str) -> int:
            """Merchant buy price from the agent context (never hard-coded)."""
            return int((prices.get(item) or {}).get("buy") or 0)

        if "trade" in opts:
            items = opts["trade"]["items"]
            seeds = sum(inv.get(k, 0) for k in ("turnip_seed", "wheat_seed"))
            coins = inv.get("coins", 0)
            seed_price = price("turnip_seed")
            if seeds < 2 and "turnip_seed" in items and seed_price:
                return pick("trade", item="turnip_seed", op="buy",
                            qty=max(1, min(3, coins // seed_price)), reason="씨앗 보충")
            surplus = sorted(((n, i) for i, n in inv.items()
                              if i in items and (prices.get(i) or {}).get("sell")
                              and n > PRODUCTION_CAP), reverse=True)
            if surplus:
                n, item = surplus[0]
                return pick("trade", item=item, op="sell", qty=min(20, n - 4),
                            reason="남는 물건 상인에게 팔기")
            # coin sink: savings above a cushion go back into goods (no runaway hoarding)
            if coins >= COIN_CUSHION:
                spend = coins - COIN_CUSHION // 2
                bread, wheat = price("bread"), price("wheat_seed")
                if "bread" in items and bread and inv.get("bread", 0) < 3:
                    return pick("trade", item="bread", op="buy",
                                qty=max(1, min(3, spend // bread)), reason="장터 빵 장보기")
                if "wheat_seed" in items and wheat and inv.get("wheat_seed", 0) < 4:
                    return pick("trade", item="wheat_seed", op="buy",
                                qty=max(1, min(4, spend // wheat)), reason="밀씨 비축")

        raw = {f: inv.get(f, 0) for f in RAW}
        if s["generosity"] >= 0.6 and sum(raw.values()) >= 10 and "store_or_take" in opts:
            item = max(RAW, key=lambda f: (raw[f], f))
            return pick("store_or_take", "chest_shared", item=item, op="store", qty=3,
                        reason="공동 창고에 나눔", expression="happy")

        if (690 <= m < 780 or 1020 <= m < 1140) and u("social") < s["sociability"] * 0.7:
            targets = [t for t in opts["visit_and_talk"]["targets"]
                       if t not in c["talked_today"]]
            if targets:
                pending = [t for t in c["unanswered"] if t in targets]
                rels = c["relationships"]
                if pending:
                    target = pending[0]
                elif u("near") < 0.5 and any(t in c["visible_actors"] for t in targets):
                    target = next(t for t in targets if t in c["visible_actors"])
                else:
                    target = min(targets, key=lambda t: (rels[t]["familiarity"] + u(t) * 0.3, t))
                topic = TOPICS[int(u("topic") * len(TOPICS))]
                return pick("visit_and_talk", target, topic=topic, reason="수다 떨고 싶음",
                            facing_cue="face_partner", expression="happy")

        if inv.get("wood", 0) < 2 and "chop_wood" in opts and u("wood") < 0.5:
            return pick("chop_wood", "forest", qty=3, reason="땔감 필요")

        wood_rich = inv.get("wood", 0) >= 6 and not any(
            o["item"] == "wood" for o in c["orders"])
        if wood_rich:  # don't strip the forest for wood nobody needs
            opts = {k: v for k, v in opts.items() if k != "chop_wood"}
        block = c.get("plan_block")
        if block and block["routine"] in ("tend_farm", "chop_wood", "forage") and (
            block["routine"] in opts
        ):
            r = block["routine"]
            return pick(r, qty=3, reason=f"계획: {block['note']}")

        works = [(WORK[k], v) for k, v in sorted(s["work_bias"].items()) if WORK[k] in opts]
        if works:
            total = sum(v for _, v in works)
            x = u("work") * total
            for r, v in works:
                x -= v
                if x <= 0:
                    return pick(r, qty=3, reason="할 일")
            return pick(works[-1][0], qty=3, reason="할 일")
        area = ("plaza", "pond", s["home_area"])[int(u("area") * 3)]
        return pick("wander", area, reason="산책", expression="neutral")

    # ------------------------------------------------------------------ Phase 2
    def _buy_food(self, c: dict[str, Any], pick: Any) -> dict[str, Any] | None:
        opts = c["options"]
        if "buy_from_shop" not in opts:
            return None
        coins = c["inventory"].get("coins", 0)
        shops = c.get("shops", {})
        best = None
        for sid in opts["buy_from_shop"]["targets"]:
            sh = shops.get(sid, {})
            for rank, food in enumerate(FOOD_BUY_PREF):
                if sh.get("stock", {}).get(food, 0) > 0 and coins >= sh["prices"].get(food, 99):
                    cand = (rank, sh["prices"][food], sid, food)
                    best = cand if best is None or cand < best else best
        if best is None:
            return None
        _, price, sid, food = best
        qty = 2 if coins >= price * 2 + 10 and shops[sid]["stock"].get(food, 0) >= 2 else 1
        return pick("buy_from_shop", sid, item=food, qty=qty, reason="가게에서 먹을 것 사기")

    def _treat(self, c: dict[str, Any], pick: Any) -> dict[str, Any] | None:
        """Spend savings at the neighbours' shops (keeps coins circulating, lowers Gini)."""
        inv = c["inventory"]
        eq = c.get("equipment", {})
        wants = []
        if "combat" in c["self"].get("skills", []) and eq.get("armor", {}).get(
            "durability", 0) <= 0:
            wants.append(("armor", 1))
        if "combat" in c["self"].get("skills", []) and inv.get("heal_potion", 0) < 2:
            wants.append(("heal_potion", 1))
        if sum(inv.get(f, 0) for f in COOKED) < 2:
            wants += [("roast_meat", 1), ("turnip_soup", 1), ("berry_stew", 1), ("bread", 2)]
        for item, q in wants:
            buy = self._buy(c, pick, item, q, "모은 돈으로 장보기")
            if buy is not None:
                return buy
        return None

    def _request_missing(self, c: dict[str, Any], pick: Any, u: Any) -> dict[str, Any] | None:
        """Post an escrowed request for a work input no known shop has in stock."""
        opts = c["options"]
        if "post_request" not in opts or u("request") > 0.25:
            return None
        if c["inventory"].get("coins", 0) < 40:
            return None
        shops = c.get("shops", {})
        recipes = c.get("recipes", {})
        inputs = sorted({i for r in recipes.values() for i in r.get("inputs", {})})
        posted = {o["item"] for o in c["orders"] if not o.get("fulfilled_by")}
        for item in inputs:
            if item in posted or item not in opts["post_request"]["items"]:
                continue
            if c["inventory"].get(item, 0) > 0:
                continue
            if any(sh.get("stock", {}).get(item, 0) > 0 for sh in shops.values()):
                continue
            return pick("post_request", item=item, qty=2, reason=f"{item} 구함 의뢰")
        return None

    def _buy(self, c: dict[str, Any], pick: Any, item: str, qty: int,
             reason: str) -> dict[str, Any] | None:
        """Buy ``item`` from the cheapest known shop that has it (P2P economy)."""
        opts = c["options"]
        if "buy_from_shop" not in opts or item not in opts["buy_from_shop"]["items"]:
            return None
        coins = c["inventory"].get("coins", 0)
        shops = c.get("shops", {})
        cands = []
        for sid in opts["buy_from_shop"]["targets"]:
            sh = shops.get(sid, {})
            n, price = sh.get("stock", {}).get(item, 0), sh.get("prices", {}).get(item, 999)
            if n > 0 and coins >= price:
                cands.append((price, sid, n))
        if not cands:
            return None
        price, sid, n = min(cands)
        q = max(1, min(qty, n, coins // price))
        return pick("buy_from_shop", sid, item=item, qty=q, reason=reason)

    def _stock(self, c: dict[str, Any], pick: Any) -> dict[str, Any] | None:
        opts = c["options"]
        if "stock_shop" not in opts:
            return None
        inv = c["inventory"]
        shop = opts["stock_shop"]["targets"][0]
        stock = c.get("shops", {}).get(shop, {}).get("stock", {})
        for item in opts["stock_shop"]["items"]:
            spare = inv.get(item, 0) - KEEP.get(item, 0)
            if spare >= 1 and stock.get(item, 0) < 12:
                return pick("stock_shop", shop, item=item, qty=min(spare, 12),
                            reason="가게 진열")
        return None

    def _craft(self, c: dict[str, Any], pick: Any, prefs: tuple[str, ...]) -> dict[str, Any] | None:
        opts = c["options"]
        for key in ("craft", "cook_meal"):
            o = opts.get(key)
            if not o:
                continue
            recipes = c.get("recipes", {})
            for r in prefs:
                out = recipes.get(r, {}).get("output", r)
                if c["inventory"].get(out, 0) >= PRODUCTION_CAP:
                    continue  # enough on hand: sell first, don't flood the market
                if r in o["items"]:
                    station = recipes.get(r, {}).get("station")
                    kinds = c.get("stations", {})
                    target = next((t for t in o["targets"] if kinds.get(t) == station),
                                  o["targets"][0])
                    return pick(key, target, item=r, qty=2, reason="생업: 만들기")
        return None

    def _profession(self, c: dict[str, Any], pick: Any, u: Any) -> dict[str, Any] | None:
        s = c["self"]
        prof = s.get("profession", "villager")
        opts = c["options"]
        inv = c["inventory"]
        coins = inv.get("coins", 0)
        shops = c.get("shops", {})
        own = shops.get(s.get("shop_id") or "", {}).get("stock", {})
        bounties = [o for o in c["orders"] if o.get("kind") == "bounty"
                    and not o.get("fulfilled_by")]
        if prof == "farmer":
            if "tend_farm" in opts:
                return pick("tend_farm", "my_plots", reason="밭일")
            if sum(inv.get(k, 0) for k in ("turnip_seed", "wheat_seed")) < 3:
                buy = self._buy(c, pick, "turnip_seed", 3, "씨앗 사기")
                if buy:
                    return buy
            return self._stock(c, pick)
        if prof == "woodcutter":
            if inv.get("wood", 0) >= 5 and own.get("charcoal", 0) < 4:
                job = self._craft(c, pick, ("charcoal",))
                if job:
                    return job
            job = self._stock(c, pick)
            if job:
                return job
            if "chop_wood" in opts and inv.get("wood", 0) < PRODUCTION_CAP * 2:
                return pick("chop_wood", "forest", qty=3, reason="벌목")
            return None
        if prof == "chef_innkeeper":
            job = self._craft(c, pick, ("roast_meat", "turnip_soup", "berry_stew"))
            if job:
                return job
            job = self._stock(c, pick)
            if job:
                return job
            if inv.get("wood", 0) < 1:
                buy = self._buy(c, pick, "wood", 3, "땔감 사기")
                if buy:
                    return buy
            if coins >= 15:
                for item, q in (("boar_meat", 2), ("turnip", 4)):
                    buy = self._buy(c, pick, item, q, "식재료 사기")
                    if buy:
                        return buy
                if "post_request" in opts and "boar_meat" in opts["post_request"]["items"] and (
                    u("req") < 0.2
                ):
                    return pick("post_request", item="boar_meat", qty=2, reason="고기 의뢰")
            if "forage" in opts:
                return pick("forage", reason="스튜용 산딸기")
            return None
        if prof == "merchant":
            if "run_stall" in opts:
                items = opts["run_stall"]["items"]
                item = items[int(u("stall") * len(items))]
                price = (c.get("item_prices", {}).get(item) or {}).get("buy") or 1
                return pick("run_stall", item=item, qty=max(1, min(6, (coins - 10) // price)),
                            reason="상단에서 물건 떼기")
            return self._stock(c, pick)
        if prof == "herbalist":
            job = self._craft(c, pick, ("heal_potion", "stamina_potion"))
            if job and (job["item"] == "heal_potion" or own.get("stamina_potion", 0) < 3):
                return job
            job = self._stock(c, pick)
            if job:
                return job
            if inv.get("herb", 0) >= 2 and inv.get("slime_gel", 0) < 1:
                buy = self._buy(c, pick, "slime_gel", 2, "슬라임 젤 사기")
                if buy:
                    return buy
            if not any(inv.get(f, 0) for f in ("wood", "charcoal")):
                buy = self._buy(c, pick, "wood", 2, "땔감 사기")
                if buy:
                    return buy
            if "gather_herb" in opts and inv.get("herb", 0) < PRODUCTION_CAP:
                return pick("gather_herb", "pond", qty=2, reason="약초 채집")
            return None
        if prof == "baker":
            job = self._craft(c, pick, ("bread",))
            if job:
                return job
            job = self._stock(c, pick)
            if job:
                return job
            if inv.get("wheat", 0) < 2:
                buy = self._buy(c, pick, "wheat", 4, "밀 사기")
                if buy:
                    return buy
            if not any(inv.get(f, 0) for f in ("wood", "charcoal")):
                buy = self._buy(c, pick, "charcoal", 2, "숯 사기") or self._buy(
                    c, pick, "wood", 2, "땔감 사기")
                if buy:
                    return buy
            return None
        if prof == "blacksmith":
            wanted = tuple(t for t in ("sword", "pickaxe", "axe", "hoe", "armor")
                           if own.get(t, 0) < 1)
            job = self._craft(c, pick, ("iron_ingot_from_goblin", *wanted))
            if job:
                return job
            job = self._stock(c, pick)
            if job:
                return job
            if "armor" in wanted and inv.get("boar_hide", 0) < 2:
                buy = self._buy(c, pick, "boar_hide", 2 - inv.get("boar_hide", 0), "가죽 사기")
                if buy:
                    return buy
            if wanted and inv.get("iron_ore", 0) < 3:
                buy = self._buy(c, pick, "iron_ore", 3, "철광석 사기") or self._buy(
                    c, pick, "goblin_ore", 1, "고블린 광석 사기")
                if buy:
                    return buy
            if wanted and not any(inv.get(f, 0) for f in ("coal", "charcoal", "wood")):
                buy = self._buy(c, pick, "coal", 2, "석탄 사기")
                if buy:
                    return buy
            return None
        if prof == "miner":
            job = self._stock(c, pick)
            if job and inv.get(job["item"], 0) >= 3:
                return job
            if "mine" in opts and inv.get("iron_ore", 0) < PRODUCTION_CAP * 2:
                return pick("mine", "quarry", qty=2, reason="채굴")
            return job
        if prof == "adventurer":
            if "sell_drops" in opts:
                return pick("sell_drops", item="all", reason="전리품 팔기")
            eq = c.get("equipment", {})
            if eq.get("weapon", {}).get("durability", 0) <= 0:
                buy = self._buy(c, pick, "sword", 1, "검 사기")
                if buy:
                    return buy
            if inv.get("heal_potion", 0) < 1 and coins >= 25:
                buy = self._buy(c, pick, "heal_potion", 1, "물약 준비")
                if buy:
                    return buy
            if "hunt" in opts:
                species = next((b["item"] for b in bounties), "any")
                return pick("hunt", "mosswood", item=species, qty=3, reason="사냥",
                            expression="happy")
            return None
        if prof == "guildmaster":
            if "post_bounty" in opts and 480 <= c["minute"] < 600:
                items = opts["post_bounty"]["items"]
                return pick("post_bounty", item=items[int(u("bounty") * len(items))],
                            reason="현상금 게시")
            return None
        return None

    def _take_food(self, c: dict[str, Any], u: Any) -> dict[str, Any] | None:
        opts = c["options"]
        if "store_or_take" not in opts:
            return None
        s = c["self"]
        items = opts["store_or_take"]["items"]
        chests = c["known_chests"]

        def food_in(cid: str) -> str | None:
            stock = chests.get(cid, {}).get("items", {})
            return next((f for f in (*COOKED, *RAW) if stock.get(f, 0) > 0 and f in items), None)

        food = food_in("chest_shared")
        if food:
            return {"routine": "store_or_take", "target": "chest_shared", "item": food,
                    "qty": 1, "op": "take", "reason": "공동 창고에서 꺼내 먹기"}
        own = food_in(s["chest_id"])
        if own:
            return {"routine": "store_or_take", "target": s["chest_id"], "item": own, "qty": 1,
                    "op": "take", "reason": "내 상자에서 꺼내기"}
        if u("steal") < s["impulsiveness"] * 0.5:
            for cid in sorted(chests):
                owner = chests[cid]["owner"]
                if owner in (s["id"], "shared", None):
                    continue
                food = food_in(cid)
                if food:
                    return {"routine": "store_or_take", "target": cid, "item": food, "qty": 1,
                            "op": "take", "reason": "너무 배고파서 몰래", "expression": "sad"}
        return None

    # ------------------------------------------------------------------ converse_turn
    def _converse_turn(self, c: dict[str, Any]) -> dict[str, Any]:
        s, p = c["self"], c["partner"]
        idx = c["turn_index"]

        def u(*salt: Any) -> float:
            return stable_unit(s["id"], p["id"], idx, len(c["turns"]), c["topic"], *salt)

        rel = p["relationship"] or {}
        valence = 0.05 + rel.get("affection", 0.0) * 0.3 + (s["sociability"] - 0.5) * 0.3
        valence += (u("v") - 0.5) * 0.5
        if c["needs"]["hunger"] >= 70:
            valence -= 0.2
        valence = round(max(-1.0, min(1.0, valence)), 3)
        emotion = ("joy" if valence > 0.4 else "warmth" if valence > 0.15 else
                   "neutral" if valence > -0.1 else
                   "worry" if c["needs"]["hunger"] >= 70 else "annoyance")
        end = (idx >= 3 and u("end") < 0.45) or idx >= c["max_turns"] - 1
        gift = "none"
        if idx <= 1 and c["giftable"] and u("gift") < s["generosity"] * 0.4:
            gift = c["giftable"][0]
        share = "none"
        rumors = [m for m in c["shareable"] if m.get("kind") == "hearsay"]
        if 1 <= idx <= 3 and rumors and u("rumor") < 0.75:
            # juicy hearsay spreads first: most confident, then most negative
            share = min(rumors, key=lambda m: (-m.get("confidence", 0), m["valence"],
                                               m["id"]))["id"]
        elif 1 <= idx <= 2 and c["shareable"] and u("share") < 0.5:
            share = min(c["shareable"], key=lambda m: (m["valence"], m["id"]))["id"]
        if idx == 0:
            utter = f"{p['name']}, {OPENERS.get(c['topic'], '요즘 어때?')}"
        elif share != "none":
            text = next(m["text"] for m in c["shareable"] if m["id"] == share)
            utter = f"그런데 그거 알아? {text[:60]}"
        elif end:
            utter = CLOSERS[int(u("close") * len(CLOSERS))]
        else:
            utter = REPLIES[int(u("reply") * len(REPLIES))]
        if gift != "none":
            utter += f" 아, 이거 {gift} 좀 가져가."
        offer = {"offer_give_item": "none", "offer_give_qty": 0, "offer_want_item": "none",
                 "offer_want_qty": 0, "trade_response": "none"}
        pend = c.get("pending_offer")
        inv = c.get("inventory", {})
        if pend:
            wanted = pend["give_item"] in c.get("wanted_items", [])
            values = c.get("item_values", {})
            fair = values.get(pend["give_item"], 1) * pend["give_qty"] >= (
                values.get(pend["want_item"], 1) if pend["want_item"] != "coins" else 1
            ) * pend["want_qty"] * 0.8
            can = inv.get(pend["want_item"], 0) - (10 if pend["want_item"] == "coins" else 0)
            ok = wanted and fair and can >= pend["want_qty"]
            offer["trade_response"] = "accept" if ok else "decline"
            utter = ("좋아, 그렇게 바꾸자!" if ok else "음, 이번엔 사양할게.") + " " + utter
        elif 1 <= idx <= 3 and not end and c.get("surplus") and u("barter") < 0.35:
            values = c.get("item_values", {})
            item = max(sorted(c["surplus"]), key=lambda i: c["surplus"][i] * values.get(i, 1))
            qty = max(1, min(4, c["surplus"][item]))
            offer.update(offer_give_item=item, offer_give_qty=qty, offer_want_item="coins",
                         offer_want_qty=max(1, values.get(item, 1) * qty))
            utter += f" 혹시 {item} {qty}개, {offer['offer_want_qty']}코인에 살래?"
        importance = 3 + (2 if gift != "none" else 0) + (2 if share != "none" else 0)
        return {"utterance": utter, "end_conversation": end, "emotion": emotion,
                "valence": valence, "importance": min(10, importance),
                "facing_cue": "face_partner", "gift_item": gift, "share_memory_id": share,
                **offer}

    # ------------------------------------------------------------------ reflect
    def _reflect(self, c: dict[str, Any]) -> dict[str, Any]:
        mems = c["memories"]
        names = c["names"]
        me = c["self"]["id"]
        top = mems[0]
        out = [{"text": f"오늘 가장 마음에 남은 일: {top['text']}",
                "importance": min(9, top["importance"] + 1), "evidence_ids": [top["id"]]}]
        counts: dict[str, list[str]] = {}
        for m in mems:
            for subj in m["subjects"]:
                if subj in names and subj != me:
                    counts.setdefault(subj, []).append(m["id"])
        facts: list[str] = []
        if counts:
            who = max(sorted(counts), key=lambda k: len(counts[k]))
            rel = c["relationships"].get(who, {})
            aff, trust = rel.get("affection", 0.0), rel.get("trust", 0.0)
            feeling = ("함께 있으면 즐겁다" if aff > 0.15 else
                       "어쩐지 믿음이 가지 않는다" if trust < -0.15 else "더 알아가고 싶다")
            out.append({"text": f"요즘 {names[who]}와(과) 자주 엮인다. {feeling}.",
                        "importance": 6, "evidence_ids": counts[who][:3]})
            if trust < -0.2:
                facts.append(f"{names[who]}은(는) 남의 물건에 손을 댈지도 모른다")
            elif aff > 0.2:
                facts.append(f"{names[who]}와(과) 사이가 좋다")
        work = [m["id"] for m in mems if m["kind"] == "observation" and m["subjects"] == [me]]
        if work:
            out.append({"text": f"오늘도 {len(work)}가지 일을 해냈다. 꾸준함이 살림을 지킨다.",
                        "importance": 5, "evidence_ids": work[:3]})
        return {"reflections": out[:3], "facts": facts[:3]}

    # ------------------------------------------------------------------ weekly goals
    def _weekly_goals(self, c: dict[str, Any]) -> dict[str, Any]:
        s = c["self"]
        refl = c["reflections"]
        names = c["names"]
        bias = s["work_bias"]
        primary = sorted(bias, key=lambda k: (-bias[k], k))[0]
        work_goal = {"farm": "밭을 잘 가꿔 마을 식량을 책임진다",
                     "wood": "마을의 장작이 떨어지지 않게 한다",
                     "forage": "숲과 연못의 먹거리를 잘 챙긴다"}[primary]
        goals = [{"text": work_goal, "evidence_ids": [refl[-1]["id"]]}]
        rels = c["relationships"]
        if rels:
            best = max(sorted(rels), key=lambda k: rels[k]["affection"])
            ev = [r["id"] for r in refl if best in r["subjects"]][:2]
            goals.append({"text": f"{names[best]}와(과) 더 가까워진다", "evidence_ids": ev})
            worst = min(sorted(rels), key=lambda k: rels[k]["trust"])
            if rels[worst]["trust"] < -0.1:
                ev = [r["id"] for r in refl if worst in r["subjects"]][:2]
                goals.append({"text": f"{names[worst]}을(를) 조심한다", "evidence_ids": ev})
        return {"goals": goals[:3]}

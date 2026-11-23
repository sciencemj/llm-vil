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

COOKED = ("turnip_soup", "berry_stew", "bread")
RAW = ("turnip", "berry")
RECIPE_PREF = ("turnip_soup", "berry_stew", "bread")
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


COIN_CUSHION = 30  # villagers keep about this much; the rest circulates at the stall


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
        primary = WORK[ranked[0]]
        secondary = WORK[ranked[1 if stable_unit(s["id"], c["day"], "plan") < 0.6 else 2]]
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
            for food in RAW:
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

        if needs["stamina"] < th["tired"]:
            return pick("rest", reason="지침", expression="tired")

        has_cooked = any(f in eat_items for f in COOKED)
        if not has_cooked and "cook_meal" in opts and (600 <= m < 720 or 990 <= m < 1080):
            items = opts["cook_meal"]["items"]
            return pick("cook_meal", item=next(r for r in RECIPE_PREF if r in items),
                        reason="끼니 준비")

        if "post_or_fulfill_order" in opts:
            return pick("post_or_fulfill_order", item=opts["post_or_fulfill_order"]["items"][0],
                        reason="주문 해결")

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
            if inv.get("wood", 0) > 6 and "wood" in items:
                return pick("trade", item="wood", op="sell", qty=inv["wood"] - 4,
                            reason="남는 장작 팔기")
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
        importance = 3 + (2 if gift != "none" else 0) + (2 if share != "none" else 0)
        return {"utterance": utter, "end_conversation": end, "emotion": emotion,
                "valence": valence, "importance": min(10, importance),
                "facing_cue": "face_partner", "gift_item": gift, "share_memory_id": share}

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

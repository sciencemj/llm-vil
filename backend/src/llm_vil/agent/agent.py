"""``Agent`` coordinator: perception, planning, routine choice, conversation, reflection.

Async ``decide_*`` methods only read state and return decisions; the engine applies them
with ``adopt_*`` on the main loop in deterministic actor order. Only ``agent/`` mutates
agent internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.protocol import Embedder, LLMRequest, LLMRouter, dumps
from ..schemas.agent import (
    AgentSpec,
    ConversationOutcome,
    ConversationTurn,
    MemoryRecord,
    PlanBlock,
    RoutineChoice,
)
from ..schemas.config import VillageConfig
from ..schemas.world import ActionResult, AtomicAction, WorldEvent, WorldView
from .conversation import ConversationSession
from .memory import RUMOR_MIN_CONFIDENCE, MemoryStream, rumor_root
from .relationships import RelationshipStore
from .replies import (
    GoalsReply,
    PlanReply,
    ReflectReply,
    RoutineReply,
    TurnReply,
    flat_schema,
)
from .routines import KnownObject, RoutineContext, next_step, resolve_targets

COOKED = ("turnip_soup", "berry_stew", "bread")
RAW_FOOD_ORDER = ("turnip", "berry")
TOPICS = ("안부", "농사", "날씨", "장터", "마을 소문", "요리", "숲")
WORK_ACTIVITIES = frozenset({"chop", "till", "plant", "water", "harvest", "forage", "cook"})
FORCED_REPLAN_AFTER = 3
TRANSCRIPT_TURNS = 6    # turns kept in a conversation memory
TURN_CHARS = 60         # per-turn clip
TRANSCRIPT_CHARS = 300  # whole transcript clip
RECENT_MEMORIES = 4     # chronological working memory shown next to retrieved ones
REJECTION_COOLDOWN = 60  # sim minutes a rejected routine is withheld from the options

SHARED_PREFIX = (
    "너는 작은 산골 마을 '{village}'에 사는 주민이다. 모든 대답은 한국어로, 지정된 JSON 형식으로만 "
    "한다. 선택지는 반드시 주어진 목록(enum) 안에서 고른다."
)
CALL_HEADS = {
    "plan_day": "오늘 하루 일정을 시간 블록으로 계획하라.",
    "choose_routine": "지금 할 일 하나(routine)를 골라라. 배고픔·기력·계획·관계를 고려하라.",
    "converse_turn": "대화 상대에게 할 다음 한마디를 정하라.",
    "reflect": "오늘의 기억을 돌아보고 2~3개의 통찰과 사실을 정리하라. 근거 기억 id를 인용하라.",
    "weekly_goals": "지난 한 주의 성찰로부터 최대 3개의 삶의 목표를 정하라. 근거 id를 인용하라.",
}

GOSSIP_TEXT = {
    "gift": "{who}이(가) {to}에게 선물을 건네는 걸 봤다",
    "traded": "{who}이(가) 장터에서 흥정하는 걸 봤다",
    "order_fulfilled": "{who}이(가) 게시판 주문으로 돈을 버는 걸 봤다",
    "harvested": "{who}의 밭에서 수확이 한창이었다",
}

EVENT_TEXT = {
    "chopped": "나무를 베어 장작을 얻었다",
    "harvested": "작물을 수확했다",
    "planted": "씨앗을 심었다",
    "tilled": "밭을 갈았다",
    "watered": "밭에 물을 주었다",
    "foraged": "산딸기를 땄다",
    "cooked": "음식을 만들었다",
    "ate": "음식을 먹었다",
    "traded": "장터에서 거래를 했다",
    "order_fulfilled": "게시판 주문을 해결하고 돈을 받았다",
    "stored": "상자에 물건을 넣었다",
    "took": "상자에서 물건을 꺼냈다",
}


@dataclass
class RoutineDecision:
    choice: RoutineChoice
    view_version: int
    target_versions: dict[str, int] = field(default_factory=dict)
    fallback: bool = False
    reason: str = ""


@dataclass
class TurnDecision:
    turn: ConversationTurn
    shared: MemoryRecord | None = None
    fallback: bool = False


class Agent:
    def __init__(self, spec: AgentSpec, config: VillageConfig, router: LLMRouter,
                 embedder: Embedder) -> None:
        self.spec = spec
        self.config = config
        self.router = router
        self.memory = MemoryStream(spec.id, embedder, config.memory, config.ticks_per_day)
        self.names = {v.id: v.name for v in config.villagers}
        self.relationships = RelationshipStore(
            spec.id, {k: v for k, v in self.names.items() if k != spec.id}
        )
        # Villagers know the village layout, but only the contents of their own chest and
        # the shared chest; other personal chests are learned by perception only.
        self.known: dict[str, KnownObject] = {}
        for o in config.objects:
            kind = config.kinds[o.kind]
            visible_items = o.kind == "chest" and o.owner in (spec.id, "shared", None)
            self.known[o.id] = KnownObject(
                id=o.id, kind=o.kind, pos=o.pos, state=o.state or kind.initial_state,
                owner=o.owner, name=o.name or o.id,
                items=dict(o.data.get("items", {})) if visible_items else {},
            )
        self.plan: list[PlanBlock] = []
        self.plan_day_index = -1
        self.routine: RoutineChoice | None = None
        self.progress: dict[str, Any] = {}
        self.recent_actions: list[str] = []
        self.rejections: list[str] = []
        self.consecutive_rejections = 0
        self.world_notes: list[str] = [
            f"내 밭: {', '.join(spec.plots)}",
            f"내 상자: {spec.chest_id}, 내 침대: {spec.bed_id}",
            "공동 창고: chest_shared (누구나 넣고 꺼낼 수 있음)",
        ]
        self.unanswered: list[str] = []
        self.goals: list[str] = []
        self.semantic_facts: list[str] = []
        self.visible_actors: list[str] = []
        self.talked_today: list[str] = []
        self.told: dict[str, list[str]] = {}  # partner -> rumor roots already shared
        self.last_reflection_day = -1  # staggered end-of-day reflection only
        self.last_threshold_reflection_tick = -1  # importance-triggered in-day reflection
        self.last_weekly_day = -1
        self.now = 0
        self.routine_cooldowns: dict[str, int] = {}
        self.counters: dict[str, int] = {
            "reflections": 0, "weekly_syntheses": 0, "conversations": 0, "hearsay_received": 0,
            "routine_fallbacks": 0, "forced_replans": 0,
        }

    @property
    def id(self) -> str:
        return self.spec.id

    def name_of(self, actor_id: str) -> str:
        return self.names.get(actor_id, actor_id)

    # ================================================================ perception
    def observe(self, view: WorldView, events: list[WorldEvent]) -> None:
        """Change-only perception: memories only for state changes and salient events."""
        tick = view.tick
        self.now = tick
        for o in view.objects:
            k = self.known.get(o.id)
            if k is None:
                self.known[o.id] = KnownObject(
                    id=o.id, kind=o.kind, pos=(o.pos.x, o.pos.y), state=o.state, owner=o.owner,
                    name=o.name, items=dict(o.data.get("items", {})), last_seen=tick,
                    version=o.version,
                )
                self.memory.perceive(tick, f"obj:{o.id}", f"새로운 {o.kind}({o.name})을 발견했다",
                                     4, subjects=[o.id])
                continue
            if k.state != o.state:
                imp = 3 if (o.owner == self.id or o.id in self.spec.plots) else 1
                if o.kind == "farm_plot" and o.state == "ready" and o.id in self.spec.plots:
                    imp = 4
                self.memory.perceive(tick, f"obj:{o.id}",
                                     f"{o.name}의 상태가 {k.state}에서 {o.state}(으)로 바뀌었다",
                                     imp, subjects=[o.id])
                k.state = o.state
            if o.kind == "chest":
                k.items = dict(o.data.get("items", {}))
            k.version = o.version
            k.last_seen = tick
        now_visible = sorted(a.id for a in view.actors)
        for a in view.actors:
            if a.id not in self.visible_actors:
                seen = self.memory.perceive(
                    tick, f"actor:{a.id}",
                    f"{a.name}이(가) 근처에 보인다 ({a.activity})", 2, subjects=[a.id],
                )
                if seen is not None and a.activity in WORK_ACTIVITIES:  # cooldown-limited
                    self.relationships.apply_event(a.id, "saw_working")
        self.visible_actors = now_visible
        for ev in events:
            self._observe_event(tick, ev)

    def observe_events(self, tick: int, events: list[WorldEvent]) -> None:
        """Events that reach the agent without a fresh view (God-mode, apply results)."""
        for ev in events:
            self._observe_event(tick, ev)

    def _observe_event(self, tick: int, ev: WorldEvent) -> None:
        me = self.id
        if ev.kind == "chest_taken":
            taker, owner = ev.actor_id or "?", ev.target_id or "?"
            item = ev.data.get("item")
            if taker == me:
                self.memory.add(tick, "observation",
                                f"{self.name_of(owner)}의 상자에서 몰래 {item}을(를) 꺼냈다", 6,
                                valence=-0.2, subjects=[owner])
            elif me in ev.witnesses:
                self.memory.add(
                    tick, "observation",
                    f"{self.name_of(taker)}이(가) {self.name_of(owner)}의 상자에서 "
                    f"{item}을(를) 가져가는 걸 봤다", 8, valence=-0.6, subjects=[taker, owner],
                )
                self.relationships.apply_event(taker, "witnessed_theft",
                                               note=f"{tick}: 상자 도둑질 목격")
        elif ev.kind == "chest_missing_discovered" and ev.target_id == me:
            missing = ", ".join(f"{m['item']} {m['qty']}개" for m in ev.data.get("missing", []))
            self.memory.add(tick, "observation", f"내 상자에서 {missing}이(가) 사라졌다!", 6,
                            valence=-0.5, subjects=[me, self.spec.chest_id])
        elif ev.kind == "utterance" and ev.actor_id != me and me not in ev.data.get("to", []):
            speaker = ev.actor_id or "?"
            self.memory.perceive(
                tick, f"utt:{speaker}",
                f"{self.name_of(speaker)}이(가) 하는 말을 들었다: \"{ev.data.get('text', '')}\"",
                2, subjects=[speaker],
            )
        elif ev.kind == "merchant_arrived":
            self.memory.perceive(tick, "merchant", "떠돌이 상인이 장터에 왔다", 3,
                                 subjects=["merchant"])
        elif ev.kind in ("weather_changed", "resource_spawned"):
            self.memory.add(tick, "external_event",
                            f"마을에 변화가 생겼다: {ev.kind} {dumps(ev.data)}", ev.importance,
                            subjects=["village"])
        elif ev.actor_id and ev.actor_id != me and ev.actor_id in self.names and (
            ev.kind in GOSSIP_TEXT
        ):
            # notable things other villagers do become first-hand gossip material
            who = ev.actor_id
            text = GOSSIP_TEXT[ev.kind].format(who=self.name_of(who),
                                               to=self.name_of(ev.target_id or ""))
            self.memory.perceive(tick, f"saw:{who}:{ev.kind}", text, 4, subjects=[who])
        elif ev.actor_id == me and ev.kind in EVENT_TEXT:
            imp = 3 if ev.kind in ("harvested", "order_fulfilled", "traded") else 2
            if ev.kind == "ate":
                imp = 1
            text = EVENT_TEXT[ev.kind]
            detail = ev.data.get("items") or ev.data.get("item") or ev.data.get("recipe") or ""
            self.memory.perceive(tick, f"self:{ev.kind}",
                                 f"{text} {dumps(detail) if detail else ''}".strip(), imp,
                                 subjects=[me])

    # ================================================================ routines
    def has_routine(self) -> bool:
        return self.routine is not None

    def next_action(self, view: WorldView) -> AtomicAction | None:
        if self.routine is None:
            return None
        self.now = view.tick
        ctx = RoutineContext(spec=self.spec, config=self.config, view=view, known=self.known)
        act = next_step(self.routine, self.progress, ctx)
        if act is None:
            self.finish_routine()
        return act

    def finish_routine(self) -> None:
        self.routine = None
        self.progress = {}

    def on_apply_result(self, result: ActionResult) -> bool:
        """Record an apply result. Returns True when a forced replan is required."""
        if result.ok:
            self.consecutive_rejections = 0
            a = result.action
            self.recent_actions = (self.recent_actions + [f"{a.kind}:{a.target_id or a.item}"])[-8:]
            return False
        return self._note_rejection(f"{result.action.kind}({result.action.target_id}): "
                                    f"{result.reason}")

    def on_world_failure(self, action: str, reason: str) -> bool:
        """A durative action failed on completion (world re-validation)."""
        return self._note_rejection(f"{action}: {reason}")

    def _note_rejection(self, text: str) -> bool:
        if self.routine is not None:
            self.routine_cooldowns[self.routine.routine] = self.now + REJECTION_COOLDOWN
        self.rejections = (self.rejections + [text])[-5:]
        self.consecutive_rejections += 1
        if "forest_reserve" in text:
            note = "숲 보호 규칙: 다 자란 나무가 너무 적으면 벨 수 없다"
            if note not in self.world_notes:
                self.world_notes.append(note)
        self.finish_routine()
        if self.consecutive_rejections >= FORCED_REPLAN_AFTER:
            self.counters["forced_replans"] += 1
            self.consecutive_rejections = 0
            return True
        return False

    # ================================================================ prompts / options
    def _persona_core(self) -> str:
        s = self.spec
        return (f"이름: {s.name} ({s.age}세, {s.occupation}). 성격: {', '.join(s.traits)}. "
                f"{s.persona}")

    def _request(self, purpose: str, view: WorldView, context: dict[str, Any],
                 schema: dict[str, Any], tier: str = "fast") -> LLMRequest:
        return LLMRequest(
            purpose=purpose, agent_id=self.id, tick=view.tick, tier=tier,  # type: ignore[arg-type]
            shared_prefix=SHARED_PREFIX.format(village=self.config.name),
            call_head=CALL_HEADS[purpose], persona_core=self._persona_core(),
            user_msg=dumps(context), response_schema=schema, context=context,
        )

    def _self_ctx(self) -> dict[str, Any]:
        s = self.spec
        return {"id": s.id, "name": s.name, "traits": s.traits, "work_bias": s.work_bias,
                "generosity": s.generosity, "impulsiveness": s.impulsiveness,
                "sociability": s.sociability, "wake_minute": s.wake_minute,
                "sleep_minute": s.sleep_minute, "bed_id": s.bed_id, "chest_id": s.chest_id,
                "home_area": s.home_area, "plots": s.plots}

    def _state_of(self, view: WorldView, oid: str) -> str | None:
        for o in view.objects:
            if o.id == oid:
                return o.state
        k = self.known.get(oid)
        return k.state if k else None

    def options(self, view: WorldView) -> dict[str, dict[str, list[str]]]:
        """Enum candidates per routine, built from the agent's *known* world only."""
        inv = view.self_actor.inventory
        cfg = self.config
        opts: dict[str, dict[str, list[str]]] = {}
        has_seed = any(inv.get(c.seed, 0) > 0 for c in cfg.crops.values())
        raining = view.weather == "rain"
        work_plots = [
            p for p in self.spec.plots
            if (st := self._state_of(view, p)) in ("ready", "untilled")
            or (st == "tilled" and has_seed) or (st == "planted" and not raining)
        ]
        if work_plots:
            opts["tend_farm"] = {"targets": ["my_plots", *work_plots]}
        mature = [k for k in self.known.values() if k.kind == "tree" and k.state == "mature"]
        if len(mature) > cfg.economy.forest_reserve:
            opts["chop_wood"] = {"targets": ["forest"]}
        bushes = sorted(k.id for k in self.known.values()
                        if k.kind == "berry_bush" and k.state == "ripe")
        if bushes:
            opts["forage"] = {"targets": bushes}
        hearths = sorted(k.id for k in self.known.values() if k.kind == "hearth")
        cookable = [
            name for name, r in cfg.recipes.items()
            if all(inv.get(i, 0) >= n + r.fuel.get(i, 0) for i, n in r.inputs.items())
            and all(inv.get(i, 0) >= n + r.inputs.get(i, 0) for i, n in r.fuel.items())
        ]
        if hearths and cookable:
            opts["cook_meal"] = {"targets": hearths, "items": cookable}
        foods = [i for i in inv if cfg.items.get(i) and cfg.items[i].food > 0]
        if foods:
            opts["eat_food"] = {"targets": ["none"], "items": sorted(foods)}
        chests = sorted(k.id for k in self.known.values() if k.kind == "chest")
        store_items = sorted(i for i, n in inv.items() if n > 0 and i != "coins")
        take_items = sorted({i for c in chests for i, n in self.known[c].items.items() if n > 0})
        if chests and (store_items or take_items):
            opts["store_or_take"] = {"targets": chests,
                                     "items": sorted(set(store_items) | set(take_items))}
        boards = sorted(k.id for k in self.known.values() if k.kind == "notice_board")
        orders = [o.id for o in view.orders if inv.get(o.item, 0) >= o.qty]
        if boards and orders:
            opts["post_or_fulfill_order"] = {"targets": boards, "items": orders}
        stalls = sorted(k.id for k in self.known.values() if k.kind == "trading_stall")
        if stalls and view.merchant_present:
            buy = [i for i, d in cfg.items.items() if d.buy and inv.get("coins", 0) >= d.buy]
            sell = [i for i, d in cfg.items.items() if d.sell and inv.get(i, 0) > 0]
            if buy or sell:
                opts["trade"] = {"targets": stalls, "items": sorted(set(buy) | set(sell))}
        others = sorted(k for k in self.names if k != self.id)
        opts["visit_and_talk"] = {"targets": others, "items": list(TOPICS)}
        opts["sleep"] = {"targets": [self.spec.bed_id]}
        opts["rest"] = {"targets": [self.spec.home_area]}
        opts["wander"] = {"targets": [a.id for a in cfg.areas]}
        for routine, until in self.routine_cooldowns.items():
            if until > view.tick and routine not in ("sleep", "rest", "wander"):
                opts.pop(routine, None)
        return opts

    def current_block(self, minute: int) -> PlanBlock | None:
        return next((b for b in self.plan if b.start_minute <= minute < b.end_minute), None)

    # ================================================================ planning
    def _default_plan(self) -> list[PlanBlock]:
        s = self.spec
        w = s.wake_minute
        return [
            PlanBlock(start_minute=w, end_minute=w + 240, routine="tend_farm", note="아침 밭일"),
            PlanBlock(start_minute=w + 240, end_minute=780, routine="forage", note="채집"),
            PlanBlock(start_minute=780, end_minute=1020, routine="chop_wood", note="오후 일"),
            PlanBlock(start_minute=1020, end_minute=s.sleep_minute, routine="wander",
                      note="저녁 산책"),
        ]

    async def decide_plan(self, view: WorldView) -> list[PlanBlock]:
        context = {
            "self": self._self_ctx(), "day": view.day, "minute": view.minute,
            "weather": str(view.weather), "goals": self.goals,
            "plots": {p: self._state_of(view, p) for p in self.spec.plots},
            "inventory": dict(view.self_actor.inventory),
            "facts": self.semantic_facts[-5:],
        }
        reply = await self.router.call(self._request("plan_day", view, context,
                                                     flat_schema(PlanReply), "deep"), PlanReply)
        if reply is None:
            return self._default_plan()
        return [PlanBlock(**b.model_dump()) for b in reply.blocks]

    def adopt_plan(self, blocks: list[PlanBlock], view: WorldView) -> None:
        self.plan = sorted(blocks, key=lambda b: b.start_minute)
        self.plan_day_index = view.day
        summary = ", ".join(f"{b.start_minute // 60:02d}시 {b.routine}" for b in self.plan)
        self.memory.add(view.tick, "plan", f"오늘의 계획: {summary}", 2, subjects=[self.id])

    async def decide_routine(self, view: WorldView, reason: str) -> RoutineDecision:
        opts = self.options(view)
        block = self.current_block(view.minute)
        rels = {
            oid: {"familiarity": r.familiarity, "affection": r.affection, "trust": r.trust}
            for oid, r in self.relationships.rels.items()
        }
        query = f"{block.routine if block else ''} {reason}"
        mems = await self.memory.aretrieve(view.tick, query, k=4)
        recent = self._recent(mems)
        context = {
            "self": self._self_ctx(), "reason": reason, "day": view.day, "minute": view.minute,
            "needs": view.self_actor.needs.model_dump(),
            "inventory": dict(view.self_actor.inventory), "weather": str(view.weather),
            "merchant_present": view.merchant_present,
            "plan_block": block.model_dump() if block else None,
            "options": opts,
            "known_chests": {k.id: {"owner": k.owner, "items": k.items}
                             for k in self.known.values() if k.kind == "chest"},
            "orders": [o.model_dump() for o in view.orders],
            "item_prices": {i: {"buy": d.buy, "sell": d.sell, "food": d.food}
                            for i, d in self.config.items.items()},
            "recent_actions": self.recent_actions, "rejections": self.rejections,
            "goals": self.goals, "relationships": rels, "unanswered": self.unanswered,
            "talked_today": self.talked_today, "visible_actors": self.visible_actors,
            "world_notes": self.world_notes, "memories": [m.text for m in mems],
            "recent_memories": [m.text for m in recent],
            "thresholds": {"hungry": self.config.needs.hungry_threshold,
                           "tired": self.config.needs.tired_threshold,
                           "forest_reserve": self.config.economy.forest_reserve},
        }
        schema = flat_schema(RoutineReply, {"routine": sorted(opts)})

        inv = view.self_actor.inventory

        def validate(r: RoutineReply) -> str | None:
            o = opts.get(r.routine)
            if o is None:
                return f"routine {r.routine} is not available now; choose from {sorted(opts)}"
            if r.routine not in ("eat_food",) and r.target not in o["targets"]:
                return f"target must be one of {o['targets']}"
            if "items" in o and r.routine != "visit_and_talk" and r.item not in o["items"]:
                return f"item must be one of {o['items']}"
            if r.routine == "store_or_take":
                return self._validate_chest_op(r, inv)
            if r.routine == "trade":
                return self._validate_trade(r, inv)
            return None

        reply = await self.router.call(self._request("choose_routine", view, context, schema),
                                       RoutineReply, validate)
        if reply is None:
            choice = self._fallback_choice(view, opts)
            return RoutineDecision(choice=choice, view_version=view.world_version,
                                   fallback=True, reason=reason)
        choice = RoutineChoice(**reply.model_dump())
        ctx = RoutineContext(spec=self.spec, config=self.config, view=view, known=self.known)
        versions = {oid: self.known[oid].version for oid in resolve_targets(choice, ctx)}
        return RoutineDecision(choice=choice, view_version=view.world_version,
                               target_versions=versions, reason=reason)

    def _validate_chest_op(self, r: RoutineReply, inv: dict[str, int]) -> str | None:
        if r.op == "store":
            have = inv.get(r.item, 0)
            if have < r.qty:
                return f"store qty {r.qty} exceeds inventory {r.item}={have}"
            return None
        if r.op == "take":
            stock = self.known[r.target].items.get(r.item, 0) if r.target in self.known else 0
            if stock < r.qty:
                return f"take qty {r.qty} exceeds known stock {r.item}={stock} in {r.target}"
            return None
        return "op must be 'store' or 'take' for store_or_take"

    def _validate_trade(self, r: RoutineReply, inv: dict[str, int]) -> str | None:
        item = self.config.items.get(r.item)
        if item is None:
            return f"unknown item {r.item}"
        if r.op == "buy":
            if item.buy is None:
                return f"{r.item} is not sold"
            cost = item.buy * r.qty
            if inv.get("coins", 0) < cost:
                return f"buying {r.qty} {r.item} costs {cost} coins; have {inv.get('coins', 0)}"
            return None
        if r.op == "sell":
            if item.sell is None:
                return f"{r.item} cannot be sold"
            if inv.get(r.item, 0) < r.qty:
                return f"sell qty {r.qty} exceeds inventory {r.item}={inv.get(r.item, 0)}"
            return None
        return "op must be 'buy' or 'sell' for trade"

    def _fallback_choice(self, view: WorldView, opts: dict[str, dict[str, list[str]]]
                         ) -> RoutineChoice:
        """Safe deterministic fallback: eat if hungry, continue the plan block, else forage."""
        m = view.minute
        if m >= self.spec.sleep_minute or m < self.spec.wake_minute - 30:
            return RoutineChoice(routine="sleep", target=self.spec.bed_id, reason="fallback")
        needs = view.self_actor.needs
        if needs.hunger >= self.config.needs.hungry_threshold and "eat_food" in opts:
            return RoutineChoice(routine="eat_food", item=opts["eat_food"]["items"][0],
                                 reason="fallback")
        block = self.current_block(m)
        if block and block.routine in opts and block.routine in ("tend_farm", "chop_wood",
                                                                   "forage"):
            return RoutineChoice(routine=block.routine, target=opts[block.routine]["targets"][0],
                                 qty=3, reason="fallback")
        if "forage" in opts:
            return RoutineChoice(routine="forage", target=opts["forage"]["targets"][0],
                                 reason="fallback")
        return RoutineChoice(routine="wander", target=self.spec.home_area, reason="fallback")

    def adopt_routine(self, decision: RoutineDecision) -> None:
        self.routine = decision.choice
        self.progress = {}
        if decision.fallback:
            self.counters["routine_fallbacks"] += 1

    # ================================================================ conversation
    def consider_invite(self, from_id: str, view: WorldView) -> bool:
        needs = view.self_actor.needs
        rel = self.relationships.get(from_id)
        if needs.hunger >= 85 or needs.stamina < 10:
            return False
        if self.routine is not None and self.routine.routine in ("sleep", "eat_food"):
            return False
        return rel is None or rel.affection > -0.6

    def _shareable(self, partner: str) -> list[MemoryRecord]:
        """Gossip worth passing on to ``partner``: credible rumors first, then notable
        first-hand sightings of a third villager. Never about the partner, never twice."""
        told = set(self.told.get(partner, []))
        rumors, sightings = [], []
        for r in self.memory.records:
            if partner in r.subjects or rumor_root(r) in told or r.source_actor_id == partner:
                continue  # not about them, not already told, never back to the teller
            about_other = any(s in self.names and s not in (self.id, partner)
                              for s in r.subjects)
            if r.kind == "hearsay" and r.confidence >= RUMOR_MIN_CONFIDENCE and (
                about_other or r.source_actor_id == "rumor"
            ):
                rumors.append(r)
            elif r.kind == "observation" and r.importance >= 4 and about_other:
                sightings.append(r)
        return (rumors[-3:] + sightings[-2:])[:4]

    def mark_told(self, partner: str, record: MemoryRecord) -> None:
        roots = self.told.setdefault(partner, [])
        root = rumor_root(record)
        if root not in roots:
            roots.append(root)
            del roots[:-50]

    def rumors(self) -> list[MemoryRecord]:
        return [r for r in self.memory.records if r.kind == "hearsay"]

    async def decide_turn(self, session: ConversationSession, view: WorldView,
                          archive_hits: list[MemoryRecord]) -> TurnDecision:
        partner = session.partner_of(self.id)
        rel = self.relationships.get(partner)
        # let retrieval follow the conversation: what the partner just said pulls in related
        # memories (instead of prescribing how to answer in the prompt)
        last_said = next((t.utterance for t in reversed(session.turns)
                          if t.speaker_id == partner), "")
        query = f"{self.name_of(partner)} {session.topic} {last_said}".strip()
        mems = await self.memory.aretrieve(view.tick, query, subjects=[partner], k=5,
                                           extra=archive_hits)
        recent = self._recent(mems)
        shareable = self._shareable(partner)
        inv = view.self_actor.inventory
        giftable = sorted(i for i, n in inv.items()
                          if n >= 2 and self.config.items.get(i) and self.config.items[i].food)
        context = {
            "self": self._self_ctx(),
            "partner": {"id": partner, "name": self.name_of(partner),
                        "relationship": rel.model_dump() if rel else None},
            "topic": session.topic, "turn_index": len(session.turns),
            "max_turns": session.max_turns,
            "turns": [{"speaker": t.speaker_id, "speaker_name": self.name_of(t.speaker_id),
                       "utterance": t.utterance} for t in session.turns],
            "memories": [{"id": m.id, "text": m.text} for m in mems],
            "recent_memories": [{"id": m.id, "text": m.text} for m in recent],
            "archive_hits": [m.text for m in archive_hits],
            "shareable": [{"id": m.id, "text": m.text, "valence": m.valence, "kind": m.kind,
                           "confidence": m.confidence} for m in shareable],
            "giftable": giftable, "needs": view.self_actor.needs.model_dump(),
            "unanswered": partner in self.unanswered,
        }
        gift_enum = ["none", *giftable]
        share_enum = ["none", *(m.id for m in shareable)]
        schema = flat_schema(TurnReply, {"gift_item": gift_enum, "share_memory_id": share_enum})

        def validate(r: TurnReply) -> str | None:
            if r.gift_item not in gift_enum:
                return f"gift_item must be one of {gift_enum}"
            if r.share_memory_id not in share_enum:
                return f"share_memory_id must be one of {share_enum}"
            return None

        reply = await self.router.call(self._request("converse_turn", view, context, schema),
                                       TurnReply, validate)
        if reply is None:
            turn = ConversationTurn(speaker_id=self.id, listener_id=partner, tick=view.tick,
                                    utterance="음, 그럼 다음에 또 이야기해요.",
                                    end_conversation=True, valence=0.1, importance=2)
            return TurnDecision(turn=turn, fallback=True)
        turn = ConversationTurn(speaker_id=self.id, listener_id=partner, tick=view.tick,
                                **reply.model_dump())
        shared = self.memory.get(reply.share_memory_id) if reply.share_memory_id != "none" else None
        return TurnDecision(turn=turn, shared=shared)

    def receive_hearsay(self, record: MemoryRecord, speaker: str, tick: int) -> MemoryRecord:
        rec = self.memory.add_hearsay(tick, record, speaker, self.name_of(speaker))
        self.counters["hearsay_received"] += 1
        if record.valence <= -0.3:
            for s in record.subjects:
                if s in self.names and s not in (self.id, speaker):
                    self.relationships.apply_event(s, "hearsay_theft", scale=rec.confidence,
                                                   note=f"{tick}: 소문을 들음")
        return rec

    def _transcript(self, turns: list[ConversationTurn]) -> str:
        """Compact ``이름: 발화`` transcript: up to 6 turns, each clipped, ~300 chars total."""
        lines = []
        for t in turns[:TRANSCRIPT_TURNS]:
            text = " ".join(t.utterance.split())
            if len(text) > TURN_CHARS:
                text = text[:TURN_CHARS - 1] + "…"
            lines.append(f"{self.name_of(t.speaker_id)}: {text}")
        out = " / ".join(lines)
        return out if len(out) <= TRANSCRIPT_CHARS else out[:TRANSCRIPT_CHARS - 1] + "…"

    def _recent(self, exclude: list[MemoryRecord], n: int = RECENT_MEMORIES) -> list[MemoryRecord]:
        """Latest ``n`` memories in time order (working memory), minus ones already retrieved,
        so a conversation that just ended is never crowded out by older high-importance ones."""
        seen = {m.id for m in exclude}
        out = [r for r in reversed(self.memory.records) if r.id not in seen][:n]
        return out[::-1]

    def apply_conversation_outcome(self, outcome: ConversationOutcome, tick: int) -> None:
        me = self.id
        partner = next(p for p in outcome.participants if p != me)
        pname = self.name_of(partner)
        if outcome.end_reason == "declined":
            if outcome.initiator_id == me:
                self.memory.add(tick, "conversation", f"{pname}에게 말을 걸었지만 거절당했다", 3,
                                valence=-0.2, subjects=[partner])
                self.relationships.apply_event(partner, "declined_invite")
            return
        their = outcome.mean_valence.get(partner, 0.0)
        mine = outcome.mean_valence.get(me, 0.0)
        importance = max((t.importance for t in outcome.turns), default=2)
        # keep what was actually said (not just the goodbye) so later retrieval finds it
        self.memory.add(
            tick, "conversation",
            f"{pname}와(과) '{outcome.topic}'에 대해 {len(outcome.turns)}마디 나눈 대화 — "
            f"{self._transcript(outcome.turns)}", importance, valence=(their + mine) / 2,
            subjects=[partner],
        )
        event = ("pleasant_chat" if their >= 0.25 else
                 "unpleasant_chat" if their <= -0.1 else "neutral_chat")
        self.relationships.apply_event(partner, event, note=f"{tick}: {outcome.topic} 대화")
        self.relationships.appraise(partner, {"affection": their * 0.08, "trust": their * 0.03,
                                              "respect": (their + mine) * 0.015})
        for giver, receiver, item in outcome.gifts:
            if receiver == me:
                self.relationships.apply_event(giver, "gift_received", note=f"{tick}: {item} 선물")
                self.world_notes = (self.world_notes + [f"{pname}에게 {item}을(를) 받았다"])[-8:]
            elif giver == me:
                self.relationships.apply_event(receiver, "gift_given")
        if outcome.turns and outcome.turns[-1].speaker_id == partner and (
            outcome.turns[-1].utterance.endswith("?") and outcome.end_reason == "max_turns"
        ):
            if partner not in self.unanswered:
                self.unanswered = (self.unanswered + [partner])[-3:]
        elif partner in self.unanswered:
            self.unanswered.remove(partner)
        if partner not in self.talked_today:
            self.talked_today.append(partner)
        self.counters["conversations"] += 1

    # ================================================================ reflection
    def wants_reflection(self) -> bool:
        return self.memory.needs_reflection()

    async def decide_reflection(self, view: WorldView, reason: str) -> ReflectReply | None:
        since = view.tick - self.config.ticks_per_day
        recent = sorted(
            (r for r in self.memory.since(since) if r.kind not in ("reflection", "goal")),
            key=lambda r: (-r.importance, -r.tick, r.id),
        )[:20]
        if not recent:
            return None
        context = {
            "self": self._self_ctx(), "reason": reason, "day": view.day,
            "memories": [{"id": r.id, "text": r.text, "importance": r.importance,
                          "subjects": r.subjects, "kind": r.kind, "valence": r.valence}
                         for r in recent],
            "relationships": {oid: r.model_dump(exclude={"recent_key_events"})
                              for oid, r in self.relationships.rels.items()},
            "names": self.names,
        }
        valid_ids = {r.id for r in recent}

        def validate(r: ReflectReply) -> str | None:
            bad = [e for it in r.reflections for e in it.evidence_ids if e not in valid_ids]
            return f"unknown evidence ids {bad[:3]}" if bad else None

        reply = await self.router.call(
            self._request("reflect", view, context, flat_schema(ReflectReply), "deep"),
            ReflectReply, validate,
        )
        if reply is None:  # fallback: one plain summary reflection
            top = recent[0]
            return ReflectReply.model_validate({
                "reflections": [{"text": f"오늘 가장 기억에 남는 일: {top.text}",
                                 "importance": 5, "evidence_ids": [top.id]}],
            })
        return reply

    def wants_weekly(self, day: int) -> bool:
        return day - self.last_weekly_day >= 7

    def adopt_reflection(self, reply: ReflectReply | None, view: WorldView,
                         mode: str = "end_of_day") -> bool:
        """Apply a reflection. ``mode`` is ``end_of_day`` or ``importance``; both may happen on
        the same day. Returns True when reflections were actually stored."""
        self.memory.importance_accum = 0
        if mode == "importance":
            self.last_threshold_reflection_tick = view.tick
        else:
            self.last_reflection_day = view.day
        if reply is None:
            return False
        for it in reply.reflections:
            ev = [e for e in it.evidence_ids if self.memory.get(e) is not None]
            subjects = sorted({s for e in ev for s in (self.memory.get(e).subjects or [])})  # type: ignore[union-attr]
            self.memory.add(view.tick, "reflection", it.text, it.importance, subjects=subjects,
                            evidence_ids=ev)
        for fact in reply.facts:
            if fact not in self.semantic_facts:
                self.semantic_facts = (self.semantic_facts + [fact])[-20:]
                self.memory.add(view.tick, "fact", fact, 5, subjects=[self.id])
        self.relationships.refresh_summaries()
        self.memory.importance_accum = 0  # facts written above never re-trigger reflection
        self.counters["reflections"] += 1
        return True

    async def decide_weekly(self, view: WorldView) -> GoalsReply | None:
        refl = self.memory.of_kind("reflection")[-14:]
        if not refl:
            return None
        context = {
            "self": self._self_ctx(), "day": view.day, "current_goals": self.goals,
            "reflections": [{"id": r.id, "text": r.text, "subjects": r.subjects} for r in refl],
            "relationships": {oid: {"affection": r.affection, "trust": r.trust}
                              for oid, r in self.relationships.rels.items()},
            "names": self.names,
        }
        valid = {r.id for r in refl}

        def validate(r: GoalsReply) -> str | None:
            bad = [e for g in r.goals for e in g.evidence_ids if e not in valid]
            return f"unknown evidence ids {bad[:3]}" if bad else None

        return await self.router.call(
            self._request("weekly_goals", view, context, flat_schema(GoalsReply), "deep"),
            GoalsReply, validate,
        )

    def adopt_weekly(self, reply: GoalsReply | None, view: WorldView) -> bool:
        self.last_weekly_day = view.day
        if reply is None:
            return False
        self.goals = [g.text for g in reply.goals][:3]  # persona core (spec) stays immutable
        for g in reply.goals[:3]:
            self.memory.add(view.tick, "goal", f"삶의 목표: {g.text}", 7, subjects=[self.id],
                            evidence_ids=g.evidence_ids)
        self.counters["weekly_syntheses"] += 1
        return True

    # ================================================================ maintenance
    def daily_maintenance(self, tick: int) -> list[MemoryRecord]:
        self.relationships.daily_decay()
        self.memory.archive_old(tick)
        self.talked_today = []
        return self.memory.drain_pending()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory": self.memory.snapshot(),
            "relationships": self.relationships.snapshot(),
            "known": {k: v.model_dump(mode="json") for k, v in self.known.items()},
            "plan": [b.model_dump(mode="json") for b in self.plan],
            "plan_day_index": self.plan_day_index,
            "routine": self.routine.model_dump(mode="json") if self.routine else None,
            "progress": self.progress,
            "recent_actions": self.recent_actions,
            "rejections": self.rejections,
            "consecutive_rejections": self.consecutive_rejections,
            "world_notes": self.world_notes,
            "unanswered": self.unanswered,
            "goals": self.goals,
            "semantic_facts": self.semantic_facts,
            "visible_actors": self.visible_actors,
            "talked_today": self.talked_today,
            "told": {k: list(v) for k, v in sorted(self.told.items())},
            "last_reflection_day": self.last_reflection_day,
            "last_threshold_reflection_tick": self.last_threshold_reflection_tick,
            "last_weekly_day": self.last_weekly_day,
            "counters": self.counters,
            "now": self.now,
            "routine_cooldowns": dict(sorted(self.routine_cooldowns.items())),
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.memory.restore(data["memory"])
        self.relationships.restore(data["relationships"])
        self.known = {k: KnownObject.model_validate(v) for k, v in data["known"].items()}
        self.plan = [PlanBlock.model_validate(b) for b in data["plan"]]
        self.plan_day_index = data["plan_day_index"]
        self.routine = RoutineChoice.model_validate(data["routine"]) if data["routine"] else None
        self.progress = dict(data["progress"])
        self.recent_actions = list(data["recent_actions"])
        self.rejections = list(data["rejections"])
        self.consecutive_rejections = data["consecutive_rejections"]
        self.world_notes = list(data["world_notes"])
        self.unanswered = list(data["unanswered"])
        self.goals = list(data["goals"])
        self.semantic_facts = list(data["semantic_facts"])
        self.visible_actors = list(data["visible_actors"])
        self.talked_today = list(data["talked_today"])
        self.told = {k: list(v) for k, v in data.get("told", {}).items()}
        self.last_reflection_day = data["last_reflection_day"]
        self.last_threshold_reflection_tick = data["last_threshold_reflection_tick"]
        self.last_weekly_day = data["last_weekly_day"]
        self.counters = dict(data["counters"])
        self.now = data["now"]
        self.routine_cooldowns = dict(data["routine_cooldowns"])

import asyncio

from llm_vil.agent.replies import PlanReply, RoutineReply, TurnReply, flat_schema
from llm_vil.llm.protocol import (
    LLMRequest,
    LLMResponse,
    LLMRouter,
    TokenBudget,
    dumps,
    extract_json,
    strip_think,
)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, texts, delay=0.0):
        self.texts = list(texts)
        self.delay = delay
        self.seen: list[LLMRequest] = []

    async def complete_json(self, req):
        self.seen.append(req)
        await asyncio.sleep(self.delay)
        return LLMResponse(text=self.texts.pop(0), model=self.model, tokens_in=10, tokens_out=5)

    async def embed(self, texts):
        return [[0.0] for _ in texts]


def req(**kw):
    return LLMRequest(purpose="choose_routine", agent_id="mina", tick=0, user_msg="{}", **kw)


def test_dumps_keeps_korean():
    s = dumps({"말": "안녕하세요"})
    assert "안녕하세요" in s and "\\u" not in s


def test_extract_json_strips_think_and_fences():
    raw = '<think>생각 {"x":1}</think>\n```json\n{"routine": "rest", "note": "a}b"}\n```'
    assert extract_json(raw) == {"routine": "rest", "note": "a}b"}
    assert strip_think("<think>x</think>ok") == "ok"


def test_flat_schemas_have_no_refs_or_unions():
    for model in (RoutineReply, PlanReply, TurnReply):
        text = dumps(flat_schema(model))
        assert "$ref" not in text and "anyOf" not in text and "oneOf" not in text
    narrowed = flat_schema(TurnReply, {"gift_item": ["none", "berry"]})
    assert narrowed["properties"]["gift_item"]["enum"] == ["none", "berry"]


async def test_router_repairs_with_validator_feedback():
    llm = ScriptedLLM(["not json", '{"routine": "fly"}', '{"routine": "rest", "target": "home"}'])
    router = LLMRouter(llm, retries=2)
    out = await router.call(req(), RoutineReply)
    assert out is not None and out.routine == "rest"
    assert router.stats.calls == 3 and router.stats.repaired == 1
    assert llm.seen[1].feedback and "[이전 응답 오류]" in llm.seen[2].prompt_text()
    assert len(router.drain_tape()) == 3


async def test_router_falls_back_after_exhausted_repairs_and_validate():
    llm = ScriptedLLM(['{"routine": "rest", "target": "x"}'] * 3)
    router = LLMRouter(llm, retries=2)
    out = await router.call(req(), RoutineReply, lambda r: "bad target")
    assert out is None and router.stats.fallbacks == 1 and router.stats.valid == 0


async def test_router_timeout_never_raises():
    router = LLMRouter(ScriptedLLM(['{"routine":"rest"}'] * 3, delay=0.2), timeout=0.01,
                       retry_base_delay_s=0)
    assert await router.call(req(), RoutineReply) is None
    assert router.stats.timeouts == 3 and router.stats.fallbacks == 1  # timeouts are retried
    assert router.obs.recent[-1].status == "fallback"


async def test_token_budget_soft_pause():
    budget = TokenBudget(per_run=20)
    router = LLMRouter(ScriptedLLM(['{"routine":"rest"}'] * 3), budget=budget)
    assert await router.call(req(), RoutineReply) is not None
    assert await router.call(req(), RoutineReply) is not None
    assert budget.paused
    assert await router.call(req(), RoutineReply) is None
    assert router.stats.budget_skips == 1

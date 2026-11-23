"""Conversation memories keep the dialogue; turns see working memory and follow the talk."""

from __future__ import annotations

from llm_vil.agent import Agent
from llm_vil.agent.conversation import ConversationSession
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import HashingEmbedder, LLMRouter
from llm_vil.schemas.agent import ConversationTurn
from llm_vil.schemas.world import AtomicAction
from llm_vil.world.world import World


class Capturing(DemoBackend):
    def __init__(self):
        super().__init__()
        self.contexts: dict[str, dict] = {}

    async def complete_json(self, req):
        self.contexts[req.purpose] = req.context
        return await super().complete_json(req)


def make(config, aid="mina"):
    llm = Capturing()
    spec = next(v for v in config.villagers if v.id == aid)
    return Agent(spec, config, LLMRouter(llm, retry_base_delay_s=0), HashingEmbedder()), llm


def session_with(turns_text: list[tuple[str, str]]) -> ConversationSession:
    s = ConversationSession(id="c1", initiator_id="mina", target_id="junho", topic="농사",
                            start_tick=0, max_turns=6)
    s.accept(0)
    for i, (who, text) in enumerate(turns_text):
        other = "junho" if who == "mina" else "mina"
        s.add_turn(ConversationTurn(speaker_id=who, listener_id=other, tick=i, utterance=text,
                                    valence=0.3), 1)
    return s


def test_outcome_memory_keeps_middle_turns(config):
    agent, _ = make(config)
    s = session_with([("mina", "준호, 밭은 잘 되어 가?"),
                      ("junho", "북쪽 밭 순무가 벌레를 먹어서 걱정이야."),
                      ("mina", "재를 뿌리면 벌레가 줄어든대!"),
                      ("junho", "좋은 생각이다. 내일 해볼게."),
                      ("mina", "그럼 이만 가볼게, 또 보자!"),
                      ("junho", "응, 잘 가!" + "!" * 200)])
    agent.apply_conversation_outcome(s.outcome(), 10)
    mem = agent.memory.of_kind("conversation")[-1]
    assert "벌레를 먹어서" in mem.text and "재를 뿌리면" in mem.text  # middle turns kept
    assert "준호: " in mem.text and "민아: " in mem.text  # Korean speaker labels
    assert len(mem.text) < 400 and mem.text.endswith("…")  # long turns clipped
    assert mem.embedding and mem.embed_model == agent.memory.embedder.model
    hit = agent.memory.retrieve(11, "순무 벌레 재", subjects=["junho"], k=1)
    assert hit[0].id == mem.id  # the dialogue content is retrievable


async def test_choose_routine_gets_deduped_recent_memories(config):
    agent, llm = make(config)
    w = World(config)
    w.apply("mina", AtomicAction(kind="wake"))
    for i in range(8):
        agent.memory.add(i, "observation", f"관찰 {i}", 9 if i < 4 else 1)
    await agent.decide_routine(w.perceive("mina"), "test")
    ctx = llm.contexts["choose_routine"]
    texts = [r.text for r in agent.memory.records]
    expected = [t for t in texts if t not in ctx["memories"]][-4:]
    assert ctx["recent_memories"] == expected  # latest 4 not retrieved, oldest -> newest
    assert not set(ctx["recent_memories"]) & set(ctx["memories"])  # deduped
    assert len(ctx["recent_memories"]) == 4


async def test_converse_turn_query_follows_partner_and_labels_speakers(config):
    agent, llm = make(config)
    w = World(config)
    w.apply("mina", AtomicAction(kind="wake"))
    agent.memory.add(0, "observation", "북쪽 숲에서 늑대 발자국을 봤다", 5, subjects=["mina"])
    for i in range(5):
        agent.memory.add(1 + i, "observation", f"평범한 일 {i}", 1)
    queries: list[str] = []
    orig = agent.memory.aretrieve

    async def spy(now, query, *a, **k):
        queries.append(query)
        return await orig(now, query, *a, **k)

    agent.memory.aretrieve = spy  # type: ignore[method-assign]
    s = session_with([("mina", "준호, 요즘 어때?"), ("junho", "숲에서 늑대 발자국을 봤어?")])
    await agent.decide_turn(s, w.perceive("mina"), [])
    assert queries[-1] == "준호 농사 숲에서 늑대 발자국을 봤어?"
    ctx = llm.contexts["converse_turn"]
    assert ctx["turns"][1] == {"speaker": "junho", "speaker_name": "준호",
                               "utterance": "숲에서 늑대 발자국을 봤어?"}
    assert any("늑대" in m["text"] for m in ctx["memories"])  # pulled by the utterance
    ids = {m["id"] for m in ctx["memories"]}
    assert ctx["recent_memories"] and not ids & {m["id"] for m in ctx["recent_memories"]}


async def test_first_turn_query_without_partner_utterance(config):
    agent, _ = make(config)
    w = World(config)
    w.apply("mina", AtomicAction(kind="wake"))
    queries: list[str] = []
    orig = agent.memory.aretrieve

    async def spy(now, query, *a, **k):
        queries.append(query)
        return await orig(now, query, *a, **k)

    agent.memory.aretrieve = spy  # type: ignore[method-assign]
    await agent.decide_turn(session_with([]), w.perceive("mina"), [])
    assert queries[-1] == "준호 농사"

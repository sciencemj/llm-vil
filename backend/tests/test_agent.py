from llm_vil.agent import Agent
from llm_vil.agent.conversation import ConversationSession
from llm_vil.engine import SimEngine
from llm_vil.llm import DemoBackend
from llm_vil.llm.protocol import HashingEmbedder, LLMRouter
from llm_vil.schemas.agent import ConversationTurn, RoutineChoice
from llm_vil.schemas.world import ActionResult, AtomicAction, Weather
from llm_vil.world.world import World


def make(config, aid="mina") -> Agent:
    spec = next(v for v in config.villagers if v.id == aid)
    return Agent(spec, config, LLMRouter(DemoBackend()), HashingEmbedder())


def test_options_come_from_known_world_and_rejections_cool_down(config):
    w = World(config)
    w.apply("mina", AtomicAction(kind="wake"))
    agent = make(config)
    view = w.perceive("mina")
    opts = agent.options(view)
    assert opts["tend_farm"]["targets"][0] == "my_plots"
    assert set(opts["tend_farm"]["targets"][1:]) <= set(agent.spec.plots)
    assert "bush_nw" in opts["forage"]["targets"]
    assert "mina" not in opts["visit_and_talk"]["targets"]
    agent.routine = RoutineChoice(routine="forage", target="bush_nw")
    agent.now = view.tick
    agent.on_apply_result(ActionResult(ok=False, actor_id="mina",
                                       action=AtomicAction(kind="forage", target_id="bush_nw"),
                                       reason="state_empty"))
    assert agent.routine is None
    assert "forage" not in agent.options(view)


def test_hearsay_lowers_trust_with_decayed_confidence(config):
    witness, listener = make(config, "mina"), make(config, "junho")
    rec = witness.memory.add(10, "observation", "도윤이 하늘의 상자에서 산딸기를 가져갔다", 8,
                             valence=-0.6, subjects=["doyun", "haneul"])
    heard = listener.receive_hearsay(rec, "mina", 20)
    assert heard.confidence == 0.7 and heard.kind == "hearsay"
    assert listener.relationships.get("doyun").trust < 0
    assert listener.relationships.get("mina").trust == 0  # speaker is not penalised


def test_conversation_outcome_updates_both_sides(config):
    a, b = make(config, "mina"), make(config, "seoyeon")
    s = ConversationSession(id="c", initiator_id="mina", target_id="seoyeon", topic="요리",
                            start_tick=0)
    s.accept(0)
    for i in range(6):
        sp = s.next_speaker
        s.add_turn(ConversationTurn(speaker_id=sp, listener_id=s.partner_of(sp), tick=i,
                                    utterance="좋아!", valence=0.6,
                                    gift_item="berry" if i == 0 else "none"), 2)
    out = s.outcome()
    a.apply_conversation_outcome(out, 12)
    b.apply_conversation_outcome(out, 12)
    assert a.relationships.get("seoyeon").familiarity > 0
    assert b.relationships.get("mina").affection > a.relationships.get("seoyeon").affection
    assert "seoyeon" in a.talked_today and a.memory.of_kind("conversation")


async def test_god_mode_hooks(config):
    e = SimEngine(config, DemoBackend())
    await e.run_until(config.start_minute + 120)
    e.set_weather(Weather.rain)
    assert e.world.weather == Weather.rain
    e.spawn_resource("berry_bush", (12, 16))
    assert "berry_bush_spawn_001" in e.world.objects
    e.inject_rumor("junho", "doyun", "도윤이 밤에 창고를 뒤졌대", confidence=0.7)
    assert e.agents["junho"].relationships.get("doyun").trust < 0
    assert e.agents["junho"].memory.of_kind("hearsay")[-1].confidence == 0.7
    e.inject_event("큰 폭풍이 다가온다")
    assert e.metrics["god_events"] == 4
    await e.run_days(0.5)
    assert e.metrics["errors"] == 0
    kinds = {d.type for d in e.drain_deltas()}
    assert {"god_event", "path_move", "clock"} <= kinds
